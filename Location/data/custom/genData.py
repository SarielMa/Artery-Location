# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 05:56:18 2021

@author: linhai
"""
import os
import json
import numpy as np
path = "data/custom/images/"
# write to train.txt
for n in os.listdir("./images"):
    with open("train.txt","a") as f:
        f.write(path+n+"\n")

lpath = "(path to)/wall_vessel_segmentation/data/train/"
for n in os.listdir("./images"):
    n = n.split(".")[0]
    with open(lpath+n+".json", "rb") as f:
        dic = json.load(f)
        wall = dic['wall']
        #wall = np.array(wall)
        f2 = open("./labels/"+n+".txt","w")
        for cc in wall:
            cc=np.array(cc)
            xm = cc[:,0].max()
            xmi= cc[:,0].min()
            ym = cc[:,1].max()
            ymi =cc[:,1].min()
            w = 2*(ym-ymi)/720
            h = 2*(xm-xmi)/160
            xc = (xm+xmi)/(2*720)
            yc = (ym+ymi)/(2*160)
            f2.write(str(0)+" "+str(xc)+" "+str(yc)+" "+str(w)+" "+str(h)+"\n")
        f2.close()