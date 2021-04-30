# -*- coding: utf-8 -*-

import pickle
import matplotlib.pyplot as plt
import os

for dim in [2,3,4,10] :
    
    fig, ax = plt.subplots(2,5)
    fig.suptitle(str(dim)+" Dimensional unimodal shells dataset")
    # index1 = -1
    
    for cur_layer in [1, 2] :
        # index1 = index1 + 1
        index2 = -1
        
        for cur_node in [10, 20, 50, 100, 200] :
            index2 = index2 + 1
            
            # ax[index1][index2].set_title("layers - "+str(cur_layer) + " nodes - "+str(cur_node))
            ax[cur_layer-1][index2].set_title("layers - "+str(cur_layer) + " nodes - "+str(cur_node))
            ax[cur_layer-1][index2].set_xlabel("Epochs")
            ax[cur_layer-1][index2].set_ylabel("error")

            
            for type_init in ["parallel", "perpen", "xavier", "parallel_norm", "perpen_norm"] :
                
                try :
                    file_name = os.path.join('.', "Res","multi_model_"+ type_init+"_"+str(dim)+"_"+str(cur_layer)+"_"+str(cur_node)+".pkl")
                    print(file_name)
                    file = open(file_name, 'rb')
                    # file = open(os.path.join('.', "Res","multi_model_"+ type_init+"_"+str(dim)+"_"+str(cur_layer)+"_"+str(cur_node)+".pkl"), 'rb')
                except :
                    # print("skipping")
                    # print(os.path.join('.', "Results", type_init+"_"+str(dim)+"_"+str(cur_layer)+"_"+str(cur_node)+".pkl"), 'rb')
                    continue
                
                res = pickle.load(file)                
                file.close()
                div = 1
                
                for i in range(4) :
                    try :
                        file_name = os.path.join('.', "Res","multi_model_"+ type_init+"_"+str(dim)+"_"+str(cur_layer)+"_"+str(cur_node)+"_"+str(i)+".pkl")
                        print(file_name)
                        file = open(file_name, 'rb')
                        div = div + 1
                        res = res + pickle.load(file) 
                    # file = open(os.path.join('.', "Res","multi_model_"+ type_init+"_"+str(dim)+"_"+str(cur_layer)+"_"+str(cur_node)+".pkl"), 'rb')
                    except :
                    # print("skipping")
                    # print(os.path.join('.', "Results", type_init+"_"+str(dim)+"_"+str(cur_layer)+"_"+str(cur_node)+".pkl"), 'rb')
                        break
                
                res = res / div
                
                if(sum(res) == 0) :
                    # print("skipping")
                    # print(os.path.join('.', "Results", type_init+"_"+str(dim)+"_"+str(cur_layer)+"_"+str(cur_node)+".pkl"), 'rb')
                    continue
                
                # print(os.path.join('.', "Results", type_init+"_"+str(dim)+"_"+str(cur_layer)+"_"+str(cur_node)+".pkl"))
                # ax[index1][index2].plot(res, label=type_init)
                ax[cur_layer-1][index2].plot(res, label=type_init)
                
            # ax[index1][index2].legend()
            ax[cur_layer-1][index2].legend()
                