from config import *
from dataset import get_ukriver_dataset
import torch
import numpy as np
from model import transformer_model
from plot import plot_preprocessed
import pickle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X,Y, _,ids = get_ukriver_dataset(preprocess = True)
X = np.asarray(X)
#X = X[:100]
X_without_categorical_features = np.delete(X, categorical_features_indices, axis=2)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
X_without_categorical_features = scaler.transform(X_without_categorical_features.reshape(-1, X_without_categorical_features.shape[-1])).reshape(X_without_categorical_features.shape)
X[:,:,not_categorical_features_indices] = X_without_categorical_features
#np.random.shuffle(X)
X = torch.from_numpy(X).float().to(device)


def calculate_prob(x, x_pred_u,x_pred_o):
    PDF = (1/torch.sqrt(2*3.14*torch.exp(x_pred_o))) * torch.exp(-0.5*((x-x_pred_u)**2)/torch.exp(x_pred_o))#[x != NAN_VALUE]
    #loss = torch.abs(x-x_pred_u) torch.sqrt(torch.exp(x_pred_o))    [x != NAN_VALUE]
    return PDF

def calcualte_categorical_prob(x,x_pred):
    categorical_prob = torch.zeros(x.shape[0], x.shape[1], len(categorical_features_indices)).to(device)
    for i,ind in enumerate(categorical_features_indices):
        #select the indices of the categorical features
       # print(x[:,:,ind].shape)
        x_pred_selected = torch.softmax(x_pred[i][:,:-1], dim = 2)
        x_pred_selected = torch.gather( x_pred_selected, dim = 2, index = x[:,:,ind].unsqueeze(2).long())
       # print(x_pred_selected)
        categorical_prob[:, :, i] = x_pred_selected.squeeze(2)
    return categorical_prob

model = transformer_model(X.shape[-1],256, X.shape[-1], ids)
model.load_state_dict(torch.load("model.pth"))
model.to(device)

epsilon = 1e-8
with torch.no_grad():
    x_pred_u, x_pred_o, x_pred_c = model(X)
    reg_prob = calculate_prob(X[:,1:,not_categorical_features_indices], x_pred_u[:,:-1,not_categorical_features_indices],x_pred_o[:,:-1,not_categorical_features_indices])
    reg_prob[X[:,1:,not_categorical_features_indices] == NAN_VALUE] = 1.0
    reg_prob_log = torch.sum(torch.log(reg_prob+epsilon), dim = 2)#[X != NAN_VALUE]
    categorical_prob = calcualte_categorical_prob((X[:,1:].long()+1),x_pred_c)
    categorical_prob[X[:,1:,categorical_features_indices] == NAN_VALUE] = 1.0
    cat_prob_log = torch.sum(torch.log(categorical_prob+epsilon), dim = 2)
    all_probs_log = reg_prob_log + cat_prob_log
   # print(all_probs_log)
  #  print(categorical_prob.shape)   
    #all_probs = torch.concatenate((reg_prob, categorical_prob), dim = 2)

    for i in range(X.shape[0]):
        plot_preprocessed(X[i,1:].cpu().numpy(),x_pred_u[i,:-1].cpu().numpy(), x_pred_o[i,:-1].cpu().numpy(), all_probs_log[i].cpu().numpy(), show = False, save_pth="newer_figures/ukriver_" + str(i) + ".svg")
