# -*- coding: utf-8 -*-

import pickle
import matplotlib.pyplot as plt
import os

for dim in [2,3,4,10] :
    
    fig, ax = plt.subplots(1,5)
    fig.suptitle("dim - "+str(dim))
    # index1 = -1
    
    for cur_layer in [1] :
        # index1 = index1 + 1
        index2 = -1
        
        for cur_node in [10, 20, 50, 100, 200] :
            index2 = index2 + 1
            
            # ax[index1][index2].set_title("layers - "+str(cur_layer) + " nodes - "+str(cur_node))
            ax[index2].set_title("layers - "+str(cur_layer) + " nodes - "+str(cur_node))

            
            for type_init in ["parallel", "perpen", "xavier", "parallel_norm", "perpen_norm"] :
                
                try :
                    file = open(os.path.join('.', "Res","multi_model_"+ type_init+"_"+str(dim)+"_"+str(cur_layer)+"_"+str(cur_node)+".pkl"), 'rb')
                except :
                    # print("skipping")
                    # print(os.path.join('.', "Results", type_init+"_"+str(dim)+"_"+str(cur_layer)+"_"+str(cur_node)+".pkl"), 'rb')
                    continue
                
                res = pickle.load(file)
                file.close()
                
                if(sum(res) == 0) :
                    # print("skipping")
                    # print(os.path.join('.', "Results", type_init+"_"+str(dim)+"_"+str(cur_layer)+"_"+str(cur_node)+".pkl"), 'rb')
                    continue
                
                print(os.path.join('.', "Results", type_init+"_"+str(dim)+"_"+str(cur_layer)+"_"+str(cur_node)+".pkl"))
                # ax[index1][index2].plot(res, label=type_init)
                ax[index2].plot(res, label=type_init)
                
            # ax[index1][index2].legend()
            ax[index2].legend()
                