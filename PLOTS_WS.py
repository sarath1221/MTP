# -*- coding: utf-8 -*-

import pickle
import matplotlib.pyplot as plt
import os
import pickle


fig, ax = plt.subplots(3,9, constrained_layout=True)
index1 = -1

for d_set in ["gaussian", "mnist"] :
  if(d_set == "gaussian") :
    temp =[True, False]
  else :
    temp =[False]

  for g_known in temp :
    index1 = index1+1
    index2 = -1

    for g_star in ["high", "low", "both"] :

      if (g_star == "both") :
        temp2 = [0.1, 0.17, 0.25, 0.5, 1, 2, 5]
      else :
        temp2 = [1]

      for C in temp2 :
        index2 = index2+1

        for model in ['WS', 'FC'] :
            
            # print(d_set+"_" +str(C)+"_"+str(g_known)+"_"+model+"_"+g_star)
            try :
                file = open(".\\Results_ws_new\\"+d_set+"_" +str(C).replace(".","")+"_"+str(g_known)+"_"+model+"_"+g_star, 'rb')
            except :
                print(d_set+"_" +str(C)+"_"+str(g_known)+"_"+model+"_"+g_star)
                continue
            res = pickle.load(file)
            file.close()
            
            ax[index1][index2].plot(res, label = model)
            
        ax[index1][index2].set_title(("g_star known" if g_known else "g_star unknown" )+" "+g_star + ((" C="+str(C)) if g_star == 'both' else ""))
        ax[index1][index2].set_xlabel("Epochs")
        ax[index1][index2].set_ylabel("error")
        lims = ax[index1][index2].get_ylim()
        ax[index1][index2].set_ylim([0, lims[1]])
        ax[index1][index2].legend()
          

          