
import torch
from config import *
from x_transformers import TransformerWrapper, Decoder

class ts_model(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, ids):
        super(ts_model, self).__init__()
        self.hidden_dim = hidden_dim    
        self.embeddings = torch.nn.ModuleList([torch.nn.Embedding(len(id)+1, hidden_dim) for id in list(ids.values())[:-1]]) #+1 needed for nans
        self.feature_embedding = torch.nn.Linear(len(not_categorical_features_indices), hidden_dim)
        self.lstm = torch.nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=2)
        self.fc_u = torch.nn.Linear(hidden_dim, output_dim)
        self.fc_o = torch.nn.Linear(hidden_dim, output_dim)
        self.categorical_layers = torch.nn.ModuleList([torch.nn.Linear( hidden_dim,len(id)+1) for id in list(ids.values())[:-1]]) #-1 because of different target

    def forward(self, x):
        #embed categorical features
        x_cat = torch.zeros(x.shape[0], x.shape[1], self.hidden_dim).to(x.device)
        for i,ind in enumerate(categorical_features_indices):
          #  print(x[:, :, ind].long())
            x_cat += self.embeddings[i](x[:, :, ind].long()+1)
        x = x_cat + self.feature_embedding(x[:,:, not_categorical_features_indices])
        #predicts all timesteps at once
        x, _ = self.lstm(x)
        u = self.fc_u(x)
        o = self.fc_o(x)
        c = []
        for i, layer in enumerate(self.categorical_layers):
            c_x = layer(x)
            c.append(c_x)
            u[:,:,categorical_features_indices[i]] = torch.argmax(c_x, dim = 2)-1
         #   o[:,:,categorical_features_indices[i]] = c_x[torch.argmax(c_x, dim = 2)]
        return u, o, c
    
class transformer_model(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, ids):
        super(transformer_model, self).__init__()
        self.hidden_dim = hidden_dim    
        self.embeddings = torch.nn.ModuleList([torch.nn.Embedding(len(id)+1, hidden_dim) for id in list(ids.values())[:-1]]) #+1 needed for nans
        self.feature_embedding = torch.nn.Linear(len(not_categorical_features_indices), hidden_dim)

        self.model = Decoder(
            dim = hidden_dim,
            depth = 6,
            heads = 4
        )
        self.fc_u = torch.nn.Linear(hidden_dim, output_dim)
        self.fc_o = torch.nn.Linear(hidden_dim, output_dim)
        self.categorical_layers = torch.nn.ModuleList([torch.nn.Linear( hidden_dim,len(id)+1) for id in list(ids.values())[:-1]]) #-1 because of different target

    def forward(self, x):
        #embed categorical features
        x_cat = torch.zeros(x.shape[0], x.shape[1], self.hidden_dim).to(x.device)
        for i,ind in enumerate(categorical_features_indices):
          #  print(x[:, :, ind].long())
            x_cat += self.embeddings[i](x[:, :, ind].long()+1)
        x = x_cat + self.feature_embedding(x[:,:, not_categorical_features_indices])
        #predicts all timesteps at once
        x = self.model(x,return_embeddings = True)
        print(x.shape)
        u = self.fc_u(x)
        o = self.fc_o(x)
        c = []
        for i, layer in enumerate(self.categorical_layers):
            c_x = layer(x)
            c.append(c_x)
            u[:,:,categorical_features_indices[i]] = torch.argmax(c_x, dim = 2)-1
         #   o[:,:,categorical_features_indices[i]] = c_x[torch.argmax(c_x, dim = 2)]
        return u, o, c