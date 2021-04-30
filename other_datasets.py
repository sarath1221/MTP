# -*- coding: utf-8 -*-


from keras.datasets import mnist
import tensorflow as tf
import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
dtype = torch.cuda.FloatTensor

classes = 10
conv_wight_size = 5
stride = 3
out_conv_features = 1
BATCH_SIZE = 100
lr = 0.001
EPOCHS = 100

nn_input = int((28-conv_wight_size)/stride) + 1


class trainData(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

def get_dataset(dset) :
    
    if(dset == "MNIST") : 
        (trainX, trainy), (testX, testy) = mnist.load_data()
        trainX = trainX - trainX.sum(axis = 0)/len(trainX)
        
        
        initX = np.expand_dims(trainX, 3)
        initX = tf.image.extract_patches(initX, (1, conv_wight_size,conv_wight_size, 1), (1, stride,stride, 1), (1,1,1,1), padding='VALID')
        initX = np.reshape(initX, (-1, conv_wight_size* conv_wight_size ))
        initX, counts = np.unique(initX, axis = 0, return_counts= True)
        counts = 1/counts
        probs = counts/sum(counts)
        
        trainX = np.expand_dims(trainX, 1)
        
        return trainX, trainy, initX, probs
    
    return None


class Classification(nn.Module):
    def __init__(self, classes, nodes = 5):
        super(Classification, self).__init__()
        
        self.conv = nn.Conv2d(1, out_conv_features, kernel_size=conv_wight_size, stride=stride)
        self.fc = nn.Linear(nn_input*nn_input*out_conv_features, nodes) 
        self.out = nn.Linear(nodes, classes)
        
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim = 1)
        
    def forward(self, x):
        
        x = self.relu(self.conv(x))
        x = self.relu(self.fc(x.reshape(-1, nn_input*nn_input*out_conv_features)))
        x = self.softmax(self.out(x))
        
        return x

def init_weights_xavier(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

def get_perpen(X1, X2, mode) :
    #print(X1)
    #print(X2)
    w = X1- X2
    mid = (X1+X2)/2
    b = - (np.sum(w*mid, axis=1))
    b = np.expand_dims(b, axis=0)
    # print(np.sum(w*mid, axis = 1) + b)
    
    if(mode.endswith("norm")) :
        norm = np.sqrt((np.sum(w**2) + b**2))/100
        w = w/norm.reshape((-1, 1))
        b = b/norm
    
    print(w)
    print(b)
    
    return w, b[0]

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
        norm = np.sqrt((np.sum(w**2) + b**2))
        w = w/norm.reshape((-1, 1))
        b = b/norm
    
    print(w)
    print(b)
    
    return w, b

def init_weights_custom(m, X, mode, probs):
    L1 = m.conv
    
    if mode.startswith('parallel') :
        X1 = np.ones((out_conv_features, conv_wight_size*conv_wight_size, conv_wight_size*conv_wight_size+1))
        for i in range(out_conv_features) :
            X1[i,:,:-1] = X[np.random.choice(len(X), conv_wight_size*conv_wight_size, replace=False, p = probs)]
        w, b = get_paral(X1, mode)
    elif mode.startswith('perpen') :
        X1 = X[np.random.choice(len(X), 2*out_conv_features, replace=False, p = probs)]
        X2 = X1[np.arange(0,len(X1), 2)]
        X1 = X1[np.arange(1,len(X1), 2)]
        w, b = get_perpen(X1, X2, mode)
    else :
        print("wrong mode")
            
    L1.weight.data = torch.tensor(np.reshape(w,(1, out_conv_features, conv_wight_size, conv_wight_size)), requires_grad=True, dtype = torch.float)
    L1.bias.data = torch.tensor(np.array(b), requires_grad=True, dtype = torch.float)
      
    init_weights_xavier(m.fc)
    init_weights_xavier(m.out)
      

def init_weights(m, X, mode, probs) :
    if mode == 'xavier':
      m.apply(init_weights_xavier)
    else :
      init_weights_custom(m, X, mode, probs)
      
      

X_train, y_train, initX, probs = get_dataset("MNIST")

for i in range(2,5) :
    
    for type_init in [ 'parallel_norm', 'xavier', 'perpen_norm', 'perpen', 'parallel'  ] :
        np.random.seed(i)
        
        train_data = trainData(torch.FloatTensor(X_train).type(dtype), torch.LongTensor(y_train).cuda())
        train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
        model = Classification(classes)
        criteria = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        init_weights(model, initX, type_init, probs)
        
        model.train()
        model.to('cuda')
        
        res = np.zeros(EPOCHS)
        acc = np.zeros(EPOCHS)
        
        for epoch in range(EPOCHS) :
            epoch_loss = 0
            epoch_acc = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criteria(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_acc = epoch_acc + sum(y_batch == (torch.argmax(y_pred, axis = 1))).item()
            
            epoch_loss = epoch_loss/len(train_loader)
            acc[epoch] = epoch_acc/len(X_train)
            res[epoch] = epoch_loss
            
            print(epoch," ",i , " ",type_init, " acc =" ,epoch_acc/len(X_train), "loss =", epoch_loss)
        
        file = open(os.path.join('.', "Res_MNIST_init", type_init+"_"+str(i)+".pkl"), 'wb')
        pickle.dump(res, file)
        pickle.dump(acc, file)
        file.close()
        
        