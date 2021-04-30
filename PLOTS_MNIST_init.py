# -*- coding: utf-8 -*-

import pickle
import matplotlib.pyplot as plt
import os

fig = plt.figure()
ax = plt.axes()
fig.suptitle("MNIST initializations")
            
for type_init in ["parallel", "perpen", "xavier", "parallel_norm", "perpen_norm"] :

    div = 0
    res = []
    acc = []
    
    for i in range(4) :
        try :
            file_name = os.path.join('.', "Res_MNIST_init", type_init+"_"+str(i)+".pkl")
            print(file_name)
            file = open(file_name, 'rb')
            div = div + 1
            if i == 0:
                res = pickle.load(file) 
                acc = pickle.load(file)
            else :
                res = res + pickle.load(file)
                acc = acc + pickle.load(file)
        except :
            print("not found ", file_name)
            break
    
    res = res / div
    acc = acc / div
    
    # print(os.path.join('.', "Results", type_init+"_"+str(dim)+"_"+str(cur_layer)+"_"+str(cur_node)+".pkl"))
    # ax[index1][index2].plot(res, label=type_init)
    ax.plot(res,  label=type_init)
    
# ax[index1][index2].legend()
ax.legend()
                