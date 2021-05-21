# -*- coding: utf-8 -*-

import pickle
import matplotlib.pyplot as plt
import os

fig = plt.figure()
ax = plt.axes()
dset_type = "MNIST"


if dset_type == "MNIST" : 
    # out_conv_features = 0
    # nodes = 100
    out_conv_features = 4
    nodes = 10
else : 
    out_conv_features = 40
    nodes = 300


fig.suptitle(dset_type+" initializations")

for type_init in ["parallel", "perpen", "xavier", "parallel_norm", "perpen_norm"] :

    div = 0
    res = 0
    acc = 0
    
    for i in range(5) :
        try :
            file_name = os.path.join('.',"Res_"+dset_type+"_init", type_init+"_same_pos_"+str(out_conv_features)+"_"+str(nodes)+"_"+str(i)+".pkl")
            print(file_name)
            file = open(file_name, 'rb')
            div = div + 1
            if i == 0:
                res = pickle.load(file) 
                acc = pickle.load(file)
            else :
                res = res + pickle.load(file)
                acc = acc + pickle.load(file)
                
            print(res[0])
        except :
            print("not found ", file_name)
            break
        
    if div != 0 :  
        
        res = res / div
        acc = acc / div
        
        # print(os.path.join('.', "Results", type_init+"_"+str(dim)+"_"+str(cur_layer)+"_"+str(cur_node)+".pkl"))
        # ax[index1][index2].plot(res, label=type_init)
    ax.plot(acc,  label=type_init)
    
# div = 0
# res = []
# acc = []
# type_init = "parallel_norm"

# for i in range(5) :
#     try :
#         file_name = os.path.join('.',"Res_"+dset_type+"_init", type_init+"_same_pos_"+str(out_conv_features)+"_"+str(nodes)+"_"+str(i)+".pkl")
#         print(file_name)
#         file = open(file_name, 'rb')
#         div = div + 1
#         if i == 0:
#             res = pickle.load(file) 
#             acc = pickle.load(file)
#         else :
#             res = res + pickle.load(file)
#             acc = acc + pickle.load(file)
#     except :
#         print("not found ", file_name)
#         break


# res = res / div
# acc = acc / div

# print(os.path.join('.', "Results", type_init+"_"+str(dim)+"_"+str(cur_layer)+"_"+str(cur_node)+".pkl"))
# ax[index1][index2].plot(res, label=type_init)
# ax.plot(res,  label=type_init, linestyle = "--")    
    
    
# ax[index1][index2].legend()
ax.legend()
                