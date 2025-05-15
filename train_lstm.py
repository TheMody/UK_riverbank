import numpy as np
import pandas as pd
from tabpfn_extensions import unsupervised
from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor
import torch
from plot import plot_locations
import matplotlib.pyplot as plt
import pickle
from config import *
from dataset import get_ukriver_dataset
from model import ts_model, baseline_model, transformer_model
from dlinear import Dlinear
import wandb
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from plot import plot_preprocessed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X,Y,_,ids = get_ukriver_dataset(preprocess = True)
X = np.asarray(X)
print("length of dataset", len(X))

#this should only scale features which are not categorical
X_without_categorical_features = np.delete(X, categorical_features_indices, axis=2)
scaler = MinMaxScaler(feature_range=(-1,1))
X_without_categorical_features = scaler.fit_transform(X_without_categorical_features.reshape(-1, X_without_categorical_features.shape[-1])).reshape(X_without_categorical_features.shape)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
X[:,:,not_categorical_features_indices] = X_without_categorical_features
#visualize each feature of x indepentenly

np.random.shuffle(X)
X_test = X[:X.shape[0]//5]
X = X[X.shape[0]//5:]
#filter X to remove rows with NaN values
# x_mean = []
# for i in range(X.shape[2]):
#     nanfilter = X[:, :, i] != NAN_VALUE
#     mean_feature = np.mean(X[:, :, i][nanfilter])
#     x_mean.append(mean_feature)
# x_mean = np.asarray(x_mean).reshape(1,1,-1)
# mean_loss = np.mean(np.abs(X - x_mean)[X != NAN_VALUE])
# print(mean_loss)
# #Y = np.asarray(Y)
# print(X.shape)

#model = ts_model(X.shape[-1],256, X.shape[-1], ids).to(device)
model = transformer_model(X.shape[-1],256, X.shape[-1], ids).to(device)
#model = baseline_model(X.shape[-1],256, X.shape[-1], ids).to(device)
#model = Dlinear(configs).to(device)

def criterion(x, x_pred_u,x_pred_o):
    loss = torch.mean(((0.5*x_pred_o + 0.5* torch.abs((x- x_pred_u))/torch.exp(x_pred_o))[x != NAN_VALUE]))
    #loss = torch.mean(((0.5*x_pred_o + 0.5* ((x- x_pred_u)**2)/torch.exp(x_pred_o))[x != NAN_VALUE]))
    return loss

def criterion_test(x,x_pred):
    loss = torch.mean(torch.abs(x-x_pred)[x != NAN_VALUE])
    return loss
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
#optimizer = SaLSA(model.parameters(),weight_decay=0.01, c = 0.5, use_mv = True,momentum=(0.0,0.0,0.99),)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)
wandb.init(project="ukriver", name="lstm")
max_epochs = 1000
batch_size = 8
i = 0
best_test_loss = 1e10
for e in range(max_epochs):
    #shuffle X along axis 0
    np.random.shuffle(X)
    for step in tqdm(range(X.shape[0]//batch_size)):
        i = step * batch_size
        if i + batch_size > X.shape[0]:
            break
        #print(i)
        x_current = torch.tensor(X[i:i+batch_size]).to(device, dtype=torch.float32)
        # def closure(backwards = False):
        #     x_pred_u, x_pred_o = model(x_current)
        #     loss = criterion(x_current[:,1:], x_pred_u[:,:-1],x_pred_o[:,:-1])
        #     if backwards: loss.backward()
        #     return loss
        x_pred_u, x_pred_o, x_pred_c = model(x_current)
        loss = criterion(x_current[:,1:,not_categorical_features_indices], x_pred_u[:,:-1,not_categorical_features_indices],x_pred_o[:,:-1,not_categorical_features_indices])
        loss_cat = 0.0
        for i,ind in enumerate(categorical_features_indices):
            loss_cat += torch.nn.functional.cross_entropy(x_pred_c[i][:,:-1,:].reshape((x_pred_c[i].shape[0]*(x_pred_c[i].shape[1]-1)), x_pred_c[i].shape[2]),(x_current[:,1:,ind].long()+1).reshape((x_pred_c[i].shape[0]*(x_pred_c[i].shape[1]-1))), ignore_index=0)

        loss = loss + loss_cat /len(categorical_features_indices)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)
       # loss = optimizer.step(closure)
        optimizer.step()
    scheduler.step()
    with torch.no_grad():
        #calculate the loss on the test set
        x_test = torch.tensor(X_test).to(device, dtype=torch.float32)
        x_pred_u, x_pred_o, x_pred_c = model(x_test)
        loss_test = criterion_test(x_test[:,1:], x_pred_u[:,:-1])
        if e % 20 == 0:
            plot_preprocessed(x_test[0,1:].cpu().numpy(),x_pred_u[0,:-1].cpu().numpy(), x_pred_o[0,:-1].cpu().numpy())
        wandb.log({"test_loss": loss_test.item(), "loss_train": loss.item(), "lr": scheduler.get_last_lr()[0]})
        print(f"Epoch {e}: loss = {loss.item()}, test_loss = {loss_test.item()}")
       # print(f"Epoch {e}: test_loss = {loss_test.item()}")
        #save model if best
        if loss_test.item() < best_test_loss:
            best_test_loss = loss_test.item()
            torch.save(model.state_dict(), "model.pth")
            print("saved model")

