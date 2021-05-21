#!/usr/bin/env python
# coding: utf-8

# In[1]:


from mpl_toolkits import mplot3d
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import datetime
import time
import os
import pickle

def display(X, Y, string, ax=None, color='darker') :
    
    Y = Y.squeeze()
    
    X1 = X[Y==0]
    X2 = X[Y==1]
    c1 = "#bb0000"
    c2 = "#0000bb"
    
    fig= None
    
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
    if color == 'lighter':
        c1 = "#ff6666"
        c2 = "#6666ff"
    
    ax.scatter(X1[:,0], X1[:,1], color = c1)
    ax.scatter(X2[:,0], X2[:,1], color = c2)
    
    if string != "" :
        ax.set_title(string)
    
    return fig, ax

def display_model(X, Y, X_points, Y_points, s) :
    
    fig, ax = display(X, Y, s)
    display(X_points, Y_points, s, ax, "lighter")
    
    fig.savefig(s)


# In[2]:

import time
import numpy as np

def get_sphere(dim, dset_type) :
    
    if dim == 2:
        n=10
    elif dim==3 :
        n=5
    elif dim==4 :
        n=2
    else:
        n=1
    
    dif = 1/(n+1.0)
    vals = np.arange(0, 1.000001, dif)
    x = np.meshgrid(*([vals]*(dim)))
    x = np.array(x).reshape((dim, -1))
    x = x[:,np.logical_or(np.any(x == 0, axis = 0), np.any(x == 1, axis = 0)) ]
    
    x = x - 0.5
    x= x / np.sqrt(np.sum((x**2), axis = 0))
    
    if(dset_type == "uni_modal") :
        if(dim in [2,3,4]) : 
            x = x[:, np.random.choice(len(x[0]), 500, replace=False)]
            
        if(dim == 10) :
            x = x[:, np.random.choice(len(x[0]), 1000, replace=False)]
    else :
        # if(dim in [2,3,4]) : 
        #     x = x[:, np.random.choice(len(x[0]), 100, replace=False)]
            
        if(dim == 10) :
            x = x[:, np.random.choice(len(x[0]), 200, replace=False)]
        
    
    return x.T


# In[3]:


def get_dataset(dim, dset_type) :
    sphere = get_sphere(dim, dset_type)
    sp1 = sphere*0.1
    sp2 = sphere*0.11
    
    dataset = []
    Y = []
    
    if dset_type == "uni_model" :
        dataset.append(sp1)
        dataset.append(sp2)
        Y.append(np.zeros(len(sp1)))
        Y.append(np.ones(len(sp2)))
        
    else :
        for i in range(dim) :
            dset = sp1.copy()
            dset[:, i] += 1
            dataset.append(dset)
            Y.append(np.zeros(len(dset)))
            
            dset = sp1.copy()
            dset[:, i] -= 1
            dataset.append(dset)
            Y.append(np.zeros(len(dset)))
            
            dset = sp2.copy()
            dset[:, i] += 1
            dataset.append(dset)
            Y.append(np.ones(len(dset)))
            
            dset = sp2.copy()
            dset[:, i] -= 1
            dataset.append(dset)
            Y.append(np.ones(len(dset)))
    
    dataset = np.concatenate(dataset, axis=0)
    Y = np.concatenate(Y, axis = 0)
    Y = Y[:, np.newaxis]
    
    return dataset, Y


# In[7]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import random


dtype = torch.cuda.FloatTensor

class trainData(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)
    
class binaryClassification(nn.Module):
    def __init__(self, layers, nodes, dim, batch_norm= False):
        super(binaryClassification, self).__init__()
        
        self.batch_norm = batch_norm

        if batch_norm :
          self.batch_norm_layers = nn.ModuleList()
        
        self.h_layers = nn.ModuleList()
        self.h_layers.append(nn.Linear(dim, nodes))
        
        if batch_norm :
          self.batch_norm_layers.append(nn.BatchNorm1d(dim))
        
        for i in range(layers-1) :
            self.h_layers.append(nn.Linear(nodes, nodes))
            # if batch_norm :
            #   self.batch_norm_layers.append(nn.BatchNorm1d(nodes))

        self.h_layers.append(nn.Linear(nodes, 1))
        self.relu = nn.ReLU()
        
    def forward(self, x):
        
        if(self.batch_norm) :
            x = self.batch_norm_layers[0](x)
            
        x = self.h_layers[0](x)
        for i in range(1, len(self.h_layers)-1) :
            x = self.relu(x)
            # if self.batch_norm :
            #   x = self.batch_norm_layers[i-1](x)
            x = self.h_layers[i](x)
        x = self.relu(x)
        x = self.h_layers[-1](x)
        
        return x

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag.squeeze() == y_test.squeeze()).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

def init_weights_xavier(m):
    if type(m) == nn.Linear :
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

def init_weights(m, X, mode) :
    if mode == 'xavier' :
      m.apply(init_weights_xavier)
    else :
      init_weights_custom(m, X, mode)

def get_perpen(X1, X2, mode) :
    #print(X1)
    #print(X2)
    w = X1- X2
    mid = (X1+X2)/2
    b = - (np.sum(w*mid, axis=1))
    b = np.expand_dims(b, axis=0)
    # print(np.sum(w*mid, axis = 1) + b)
    
    if(mode.endswith("norm")) :
        norm = np.sqrt((np.sum(w**2, axis=1) + b**2))
        w = w/norm.reshape((-1,1))
        b = b/norm
    
    return w, np.squeeze(b)

def get_paral(X1, mode) :
    #print(X1)
    #print(X2)
    # w = X1 + X2
    # mid = (X1+X2)/2
    # b = - (np.sum(w*mid, axis=1))
    # print(np.sum(w*mid, axis = 1) + b)
    
    w = np.zeros((len(X1), X1.shape[1]))
    b = np.zeros(len(X1))
    
    for i in range(len(X1)):
        for j in range( X1.shape[1]):
            w[i, j] = ((-1)**j)*np.linalg.det(np.delete(X1[i], j, axis = 1))
    
        b[i] = ((-1)**X1.shape[1])*np.linalg.det(np.delete(X1[i], X1.shape[1], axis = 1))
        
    if(mode.endswith("norm")) :
        norm = np.sqrt((np.sum(w**2, axis=1) + b**2))
        w = w/norm.reshape((-1,1))
        b = b/norm
    
    print(w)
    print(b)
    
    return w, b

def display_W(W, b, ax, fig) :
    
    x_lim = np.array(ax.get_xlim())
    for i in range(len(W)) :
        y_vals = (-b[i]-W[i][0]*x_lim)/W[i][1]
        ax.plot(x_lim, y_vals, '--')
        

def pass_through(vals, layer) :
    
    res = np.add(np.matmul(vals, layer.weight.data.T), layer.bias.data).numpy()
    # res[res<0] = 0
    
    return res

def get_pca(X1, mode) :
    
    w = np.zeros((len(X1), X1.shape[2]))
    b = np.zeros(len(X1))
    
    for i in range(len(X1)):
        pca = PCA()
        pca.fit(X1[i])
        w[i] = pca.components_[0]
    
# if(mode.endswith("norm")) :
    norm = np.sqrt((np.sum(w**2, axis=1) + b**2))
    w = w/norm.reshape((-1,1))
    # b = b/norm
    
    print(w)
    print(b)
    
    return w, b    

def init_weights_custom(m, X, mode):
    L1 = m.h_layers[0]
    
    if mode.startswith('parallel') :
        X1 = np.ones((L1.out_features, dim, dim+1))
        for i in range(L1.out_features) :
            X1[i,:,:-1] = X[np.random.choice(len(X), dim, replace=False)]
        w, b = get_paral(X1, mode)
    elif mode.startswith('perpen') :
        X1 = X[np.random.choice(len(X), 2*L1.out_features, replace=True)]
        X2 = X1[np.arange(0,len(X1), 2)]
        X1 = X1[np.arange(1,len(X1), 2)]
        w, b = get_perpen(X1, X2, mode)
    else :
        print("wrong mode")
            
    L1.weight.data = torch.tensor(w, requires_grad=True, dtype = torch.float)
    L1.bias.data = torch.tensor(b, requires_grad=True, dtype = torch.float)

    if(len(m.h_layers) > 2) :
      # # print("yeah")
      
        
         L2 = m.h_layers[1]
      
      # # CHANGE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
         init_weights_xavier(L2)
      
         # if mode.startswith('parallel') :
             
         #      # X1 = np.ones((L2.out_features,  min(2*L2.out_features, len(X)), L1.out_features))
         #      # for i in range(L2.out_features) :
         #      #     X1[i,:,:] = pass_through(X[np.random.choice(len(X),  min(2*L2.out_features, len(X)), replace=False)], L1)
         #      # w, b = get_pca(X1, mode)

         #       X1 = np.ones((L2.out_features,  L1.out_features, L1.out_features+1))
         #       for i in range(L2.out_features) :
         #           X1[i,:,:-1] = pass_through(X[np.random.choice(len(X),  L1.out_features, replace=False)], L1)
         #       w, b = get_paral(X1, mode)     
                
         # elif mode.startswith('perpen') :
         #     X1 = pass_through(X[np.random.choice(len(X), 2*L2.out_features, replace=False)], L1)
         #     X2 = X1[np.arange(0,len(X1), 2)]
         #     X1 = X1[np.arange(1,len(X1), 2)]
         #     w, b = get_perpen(X1, X2, mode)
         # else :
         #     print("wrong mode")
                
         # L2.weight.data = torch.tensor(w, requires_grad=True, dtype = torch.float)
         # L2.bias.data = torch.tensor(b, requires_grad=True, dtype = torch.float)      
        
         # init_weights_xavier(L2)

lr=0.001
layers = [1, 2]
nodes = [10,20,50, 100,200]
# layers = [ 1]
# nodes = [100, 200]
BATCH_SIZE = 8000
Max_epochs = 70000
batch_norm=False

epochs_dict = {2 : {1 : {10 : 70000,
                         20 : 70000,
                         50 : 70000,
                         100 : 70000,
                         200 : 70000},
                    2 : {10 : 30000,
                         20 : 30000,
                         50 : 12000,
                         100 : 10000,
                         200 : 5000}},
               3 : {1 : {10 : 40000,
                         20 : 30000,
                         50 : 60000,
                         100 : 60000,
                         200 : 60000},
                    2 : {10 : 50000,
                         20 : 60000,
                         50 : 12000,
                         100 : 8000,
                         200 : 5000}},
               4 : {1 : {10 : 30000,
                         20 : 30000,
                         50 : 30000,
                         100 : 30000,
                         200 : 60000},
                    2 : {10 : 60000,
                         20 : 60000,
                         50 : 17000,
                         100 : 10000,
                         200 : 5000}},
               10: {1 : {10 : 60000,
                         20 : 70000,
                         50 : 70000,
                         100 : 70000,
                         200 : 70000},
                    2 : {10 : 30000,
                         20 : 40000,
                         50 : 50000,
                         100 : 17000,
                         200 : 10000}}}



# n=1000 for 2 dim
# n=10 for 3 dim
# n=3 for 4 dim
# n=1 for 10 dim


for i in [1,2,3,4] :

    for dset_type in ["multi_model"]:
    
        for type_init in ['perpen_norm'] :
        # for type_init in ['parallel_norm'] :
        
            for dim  in [4] :    
        
                X_train, y_train = get_dataset(dim, dset_type)
                    
                for cur_layer in layers :
                    
                   for cur_node in nodes :
                       
                        np.random.seed(i)
                        
                        print(dset_type+"_"+type_init+"_"+str(dim)+"_"+str(cur_layer)+"_"+str(cur_node))
                
                        train_data = trainData(torch.FloatTensor(X_train).type(dtype), torch.FloatTensor(y_train).type(dtype))
                        train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
                        model = binaryClassification(cur_layer, cur_node, dim, batch_norm)
                        criterion = nn.BCEWithLogitsLoss()
                        optimizer = optim.Adam(model.parameters(), lr=lr)
        
                        init_weights(model, X_train, type_init)
        
                        model.train()
                        count = 0
                        
                        model.to('cuda')
                        Epochs = -1
                        epoch_acc = 0
                        Max_epochs = epochs_dict[dim][cur_layer][cur_node]
                        
                        results = np.zeros(Max_epochs+1)
                        time_start = time.time()
                        
                        while Epochs < Max_epochs  :
                            
                            Epochs = Epochs+1
                            epoch_loss = 0
                            epoch_acc = 0
        
                            # if Epochs%100 == 0 :
                            #     print(f'Epoch {Epochs+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}', end = '\n')
                            #     fig, ax = display(X_train, y_train, Epochs)
                            #     display_W(model.h_layers[0].weight.data, model.h_layers[0].bias.data, ax, fig)
                            #     ax.set_ylim([-1.5, 1.5])
                            #     # fig.savefig((".//Custom//" )+str(Epochs))
                            #     # plt.close(fig)                    
        
        
                            
                            for X_batch, y_batch in train_loader:
                                optimizer.zero_grad()
                                
                                y_pred = model(X_batch)
                                
                                loss = criterion(y_pred, y_batch)
                                acc = binary_acc(y_pred, y_batch)
                                
                                loss.backward()
                                optimizer.step()
                                
                                epoch_loss += loss.item()
                                epoch_acc += acc.item()
                                
                            epoch_loss = epoch_loss/len(train_loader)
                            epoch_acc = epoch_acc/len(train_loader)
                            results[Epochs] = epoch_loss
                            
                            if(Epochs%500 == 0) : 
                                print(i, Epochs, epoch_loss, type_init, dim, cur_layer, cur_node)
                        
                        
                        ## CHANGE!!!!!!!!!!!!!!!!!!!!!!!
                        if not batch_norm :
                            file = open(os.path.join('.', "Res_f", dset_type+"_"+type_init+"_"+str(dim)+"_"+str(cur_layer)+"_"+str(cur_node)+"_"+str(i)+".pkl"), 'wb')
                        else :
                            file = open(os.path.join('.', "Res_f", dset_type+"_"+type_init+"_"+str(dim)+"_"+str(cur_layer)+"_"+str(cur_node)+"_"+str(i)+"_batch_norm.pkl"), 'wb')
                        pickle.dump(results, file)
                        file.close()
                        
                        time_end = time.time()
                        print(datetime.timedelta(seconds = time_end - time_start))

# train_acc = epoch_acc
# print(layers[cur_layer], " ", nodes[cur_node], " ", Epochs, " ",  train_acc )
        
#        results_train.append([Epochs, train_acc])

#file = open("res.pkl", 'wb')
#pickle.dump(results_train, file)
#file.close()


