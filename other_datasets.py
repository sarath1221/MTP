# -*- coding: utf-8 -*-


from keras.datasets import mnist
import tensorflow as tf
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
dtype = torch.cuda.FloatTensor


classes = None
in_channels = None
out_conv_features = None
conv_wight_size = None
nodes = None
stride = None
BATCH_SIZE = None
lr = None
EPOCHS = None


class trainData(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

def get_dataset() :
    
    global classes
    global in_channels
    global out_conv_features
    global conv_wight_size
    global nodes
    global stride
    global BATCH_SIZE
    global lr
    global EPOCHS
    global nn_input
    global dset_type
    
    global counts_each
    global counts_pos
    global init_vals
    
    probs = None
    initX = None
    
    if(dset_type == "MNIST") : 
        
        classes = 10
        in_channels = 1
        out_conv_features = 4
        conv_wight_size = 5
        nodes = 10
        stride = 3
        BATCH_SIZE = 100
        lr = 0.001
        EPOCHS = 100
        nn_input = int((28-conv_wight_size)/stride) + 1
        
        # classes = 10
        # in_channels = 1
        # out_conv_features = 0
        # conv_wight_size = 0
        # nodes = 100
        # stride = 1
        # BATCH_SIZE = 100
        # lr = 0.001
        # EPOCHS = 100
        # nn_input = int((28-conv_wight_size)/stride) + 1
        
        (trainX, trainy), (testX, testy) = mnist.load_data()
        mean = trainX.sum(axis = 0)/len(trainX)
        trainX = trainX - mean
        testX = testX - mean
        
        # trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1]*trainX.shape[2]))
        # initX = trainX
        
        initX = np.expand_dims(trainX, 3)
        initX = tf.image.extract_patches(initX, (1, conv_wight_size,conv_wight_size, 1), (1, stride,stride, 1), (1,1,1,1), padding='VALID')
        initX = np.reshape(initX, (-1, conv_wight_size* conv_wight_size ))
        initX, counts = np.unique(initX, axis = 0, return_counts= True)
        counts = 1/counts
        probs = counts/sum(counts)
        

        
        trainX = np.expand_dims(trainX, 1)
        testX = np.expand_dims(testX, 1)
        
    elif(dset_type == "CIFAR") :
        
        classes = 10
        in_channels = 3
        out_conv_features = 40
        conv_wight_size = 5
        nodes = 300
        stride = 3
        BATCH_SIZE = 10000
        lr = 0.0001
        EPOCHS = 1000
        nn_input = int((32-conv_wight_size)/stride) + 1
        
        (trainX, trainy), (testX, testy) = tf.keras.datasets.cifar10.load_data()   # 32 X 32 X 3
        trainy = np.squeeze(trainy)
        testy = np.squeeze(testy)
        
        mean  = trainX.sum(axis = 0)/len(trainX)
        
        trainX = trainX - mean
        testX = testX - mean
        
        initX = tf.image.extract_patches(trainX, (1, conv_wight_size,conv_wight_size, 1), (1, stride,stride, 1), (1,1,1,1), padding='VALID')
        
        
        initX = np.reshape(initX, (-1, conv_wight_size* conv_wight_size*3 ))       
        initX = np.reshape(initX, (-1, conv_wight_size, conv_wight_size, 3))       # Conv_size X Conv_size X 3
        initX = np.moveaxis(initX, -1, 1)                                           # 3 X Conv_size X Conv_size
        initX = np.reshape(initX, (-1, conv_wight_size* conv_wight_size*3 ))        # reshape
        initX, counts = np.unique(initX, axis = 0, return_counts= True)
        counts = 1/counts
        probs = counts/sum(counts)        
        
        # initX = np.reshape(initX, (initX.shape[0], initX.shape[1] * initX.shape[2],  conv_wight_size* conv_wight_size*3 ))  
        # initX = np.reshape(initX, (initX.shape[0], initX.shape[1], conv_wight_size, conv_wight_size, 3))
        # initX = np.moveaxis(initX, -1, 2)
        # initX = np.reshape(initX, (initX.shape[0], initX.shape[1], conv_wight_size* conv_wight_size*3 ))
        
        # counts_each = [None] * initX.shape[1]
        # counts_pos = [None] * initX.shape[1]
        # init_vals = [None] * initX.shape[1]
        
        # for i in range(initX.shape[1]) :
        #     init_vals[i], counts_each[i] = np.unique(initX[:,i,:], axis = 0, return_counts = True)
        #     counts_pos[i] = len(counts_each[i])
        #     counts_each[i] = (1/counts_each[i])/(sum(1/counts_each[i]))
            
        # counts_pos = np.array(counts_pos) / sum(counts_pos)
        
        
        
        trainX = np.moveaxis(trainX, -1, 1)                                         # 3 X 32 X 32
        testX = np.moveaxis(testX, -1, 1)
        

        
    return trainX, trainy, initX, probs, testX, testy
    

class Classification(nn.Module):
    def __init__(self, classes):
        super(Classification, self).__init__()
        
        # if(dset_type == "CIFAR") :
        self.conv = nn.Conv2d(in_channels, out_conv_features, kernel_size=conv_wight_size, stride=stride)
        self.fc = nn.Linear(nn_input*nn_input*out_conv_features, nodes) 
        self.out = nn.Linear(nodes, classes)
            
        # elif(dset_type == "MNIST") :
        #     self.fc1 = nn.Linear(784, nodes) 
        #     self.fc2 = nn.Linear(nodes, nodes)
        #     self.out = nn.Linear(nodes, classes)
            
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim = 1)
        
    def forward(self, x):
        
        
        # if(dset_type == "CIFAR") :
        x = self.relu(self.conv(x))
        x = self.relu(self.fc(x.reshape(-1, nn_input*nn_input*out_conv_features)))
        x = self.softmax(self.out(x))
            
        # elif(dset_type == "MNIST") :
        #     x = self.relu(self.fc1(x))
        #     x = self.relu(self.fc2(x))
        #     x = self.softmax(x)
        
        return x

def init_weights_xavier(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight, gain = np.sqrt(2))
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
    
    print(w.shape)
    print(b.shape)
    
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
    
    print(w.shape)
    print(b.shape)
    
    return w, b

def init_weights_custom(m, X, mode, probs):
    
    # 
    # if dset_type == "CIFAR" :
        L1 = m.conv
    
        if mode.startswith('parallel') :
            
            
            X1 = np.ones((out_conv_features, conv_wight_size*conv_wight_size*in_channels, conv_wight_size*conv_wight_size*in_channels+1))
            for i in range(out_conv_features) :
                X1[i,:,:-1] = X[np.random.choice(len(X), conv_wight_size*conv_wight_size*in_channels, replace=False, p = probs)]
                
            # X1 = np.ones((out_conv_features, conv_wight_size*conv_wight_size*in_channels, conv_wight_size*conv_wight_size*in_channels+1))
            # for i in range(out_conv_features) :
            #     temp = np.random.choice(range(len(counts_pos)), 1, p = counts_pos)[0]
            #     X1[i,:,:-1] = init_vals[temp][np.random.choice(len(counts_each[temp]), conv_wight_size*conv_wight_size*in_channels, replace=False, p = counts_each[temp])]
                
            w, b = get_paral(X1, mode)
            
        elif mode.startswith('perpen') :
            
            X1 = X[np.random.choice(len(X), 2*out_conv_features, replace=False, p = probs)]
            X2 = X1[np.arange(0,len(X1), 2)]
            X1 = X1[np.arange(1,len(X1), 2)]
            w, b = get_perpen(X1, X2, mode)
            
            # X1 = np.ones((out_conv_features, conv_wight_size*conv_wight_size*in_channels))
            # X2 = np.ones((out_conv_features, conv_wight_size*conv_wight_size*in_channels))
            
            # for i in range(out_conv_features) :
            #     temp = np.random.choice(range(len(counts_pos)), 1, p = counts_pos)[0]
            #     X1[i], X2[i] = init_vals[temp][np.random.choice(len(counts_each[temp]), 2, replace=False, p = counts_each[temp])]
            
            # w, b = get_perpen(X1, X2, mode)
            
        else :
            print("wrong mode")
            

        L1.weight.data = torch.tensor(np.reshape(w,(out_conv_features, in_channels, conv_wight_size, conv_wight_size)), requires_grad=True, dtype = torch.float)
        L1.bias.data = torch.tensor(np.array(b), requires_grad=True, dtype = torch.float)
        
        init_weights_xavier(m.fc)
        init_weights_xavier(m.out)



    # elif(dset_type == "MNIST") :
    #     L1 = m.fc1
    
    #     if mode.startswith('parallel') :
    #         X1 = np.ones((L1.out_features, 784, 784+1))
    #         for i in range(L1.out_features) :
    #             X1[i,:,:-1] = X[np.random.choice(len(X), 784, replace=False)]
    #         w, b = get_paral(X1, mode)
    #     elif mode.startswith('perpen') :
    #         X1 = X[np.random.choice(len(X), 2*L1.out_features, replace=False)]
    #         X2 = X1[np.arange(0,len(X1), 2)]
    #         X1 = X1[np.arange(1,len(X1), 2)]
    #         w, b = get_perpen(X1, X2, mode)
    #     else :
    #         print("wrong mode")
# 
        # L1.weight.data = torch.tensor(w, requires_grad=True, dtype = torch.float)
        # L1.bias.data = torch.tensor(b, requires_grad=True, dtype = torch.float)  
        
        # init_weights_xavier(m.fc2)
        # init_weights_xavier(m.out)       

      

      

def init_weights(m, X, mode, probs) :
    if mode == 'xavier':
      m.apply(init_weights_xavier)
    else :
      init_weights_custom(m, X, mode, probs)
      
      
dset_type = "CIFAR"
X_train, y_train, initX, probs, testX, testy = get_dataset()


for i in range(5) :
    
    for type_init in ["parallel_norm" ] :
        
        np.random.seed(i)
        
        train_data = trainData(torch.FloatTensor(X_train).type(dtype), torch.LongTensor(y_train).cuda())
        test_data = trainData(torch.FloatTensor(testX).type(dtype), torch.LongTensor(testy).cuda())
        
        train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
        test_loader =  DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)
        
        model = Classification(classes)
        criteria = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        init_weights(model, initX, type_init, probs)
        
       
        model.to('cuda')
        
        res = np.zeros(EPOCHS)
        acc = np.zeros(EPOCHS)
        
        for epoch in range(EPOCHS) :
            train_loss = 0
            test_acc = 0
            
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criteria(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            model.eval()
            for X_batch, y_batch in test_loader:
                y_pred = model(X_batch)
                test_acc = test_acc + sum(y_batch == (torch.argmax(y_pred, axis = 1))).item()
               
            
            train_loss = train_loss/len(train_loader)
            
            acc[epoch] = test_acc/len(test_loader.dataset)
            res[epoch] = train_loss
            
            print(epoch," ",i , " ",type_init, " acc =" ,test_acc/len(test_loader.dataset), "loss =", train_loss)
        
        #CHANGE IT
        # file = open(os.path.join('.', "Res_"+dset_type+"_init", type_init+"_"+str(out_conv_features)+"_"+str(nodes)+"_"+str(i)+".pkl"), 'wb')
        file = open(os.path.join('.', "Res_"+dset_type+"_init", type_init+"_same_pos_"+str(out_conv_features)+"_"+str(nodes)+"_"+str(i)+".pkl"), 'wb')
        pickle.dump(res, file)
        pickle.dump(acc, file)
        file.close()
        
        
        
        