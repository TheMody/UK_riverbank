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
from model import ts_model
import wandb
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X,Y,_,_ = get_ukriver_dataset(preprocess = True)
X = np.asarray(X)
np.random.shuffle(X)
X_test = X[:X.shape[0]//5]
X = X[X.shape[0]//5:]
#Y = np.asarray(Y)
print(X.shape)
#visualize each feature of x indepentenly
# import matplotlib.pyplot as plt 
# for i in range(X.shape[2]):
#     print( X[1,:,i])
#     plt.plot(range(100),X[1,:,i])
#     plt.title(all_features[i])
#     plt.show()

model = ts_model(X.shape[-1],256, X.shape[-1]).to(device)

def criterion(x, x_pred_u,x_pred_o):
    loss = torch.mean(((0.5*x_pred_o + 0.5* ((x- x_pred_u)**2)/torch.exp(x_pred_o))[x != NAN_VALUE]))
    return loss

def criterion_test(x,x_pred):
    loss = torch.mean(torch.abs(x-x_pred)[x != NAN_VALUE])
    return loss
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

wandb.init(project="ukriver", name="lstm")
max_epochs = 10000
batch_size = 8
i = 0
for e in range(max_epochs):
    #shuffle X along axis 0
    np.random.shuffle(X)
    for step in tqdm(range(X.shape[0]//batch_size)):
        i = step * batch_size
        if i + batch_size > X.shape[0]:
            break
        #print(i)
        x_current = torch.tensor(X[i:i+batch_size]).to(device, dtype=torch.float32)
        x_pred_u, x_pred_o = model(x_current)
        # print(x_pred.shape)
        # print(x_current.shape)
        loss = criterion(x_current[:,1:], x_pred_u[:,:-1],x_pred_o[:,:-1])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    #calculate the loss on the test set
    x_test = torch.tensor(X_test).to(device, dtype=torch.float32)
    x_pred_u, x_pred_o = model(x_test)
    loss_test = criterion_test(x_test[:,1:], x_pred_u[:,:-1])
    wandb.log({"test_loss": loss_test.item(), "loss": loss.item()})
    print(f"Epoch {e}: loss = {loss.item()}, test_loss = {loss_test.item()}")


