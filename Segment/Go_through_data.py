# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 16:47:02 2021

@author: linhai
"""
import os
import pydicom
import glob
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import torch
from skimage.draw import polygon2mask
import torchvision
import random
import json
from PIL import Image as im

def readDicom(path):
    pi = os.path.basename(path).split('_')[1]
    dcm_size = len(glob.glob(path+'/*.dcm'))
    dcms = [path+'/E'+pi+'S101I%d.dcm'%dicom_slicei for dicom_slicei in range(1,dcm_size+1)]
       #dcm_f = pydicom.read_file(dcms[0]).pixel_array
    dcm_size = 720
    dcm_img = np.zeros((dcm_size,dcm_size,len(dcms)))
    for dcmi in range(len(dcms)):
        cdcm = pydicom.read_file(dcms[dcmi]).pixel_array
        dcm_img[dcm_size//2-cdcm.shape[0]//2:dcm_size//2+cdcm.shape[0]//2,
                dcm_size//2-cdcm.shape[1]//2:dcm_size//2+cdcm.shape[1]//2,dcmi] = cdcm
        
    return dcm_img
        
def listContourSlices(qvsroot, dcm_img):
    avail_slices = []
    qvasimg = qvsroot.findall('QVAS_Image')
    for dicom_slicei in range(dcm_img.shape[2]):
        conts = qvasimg[dicom_slicei - 1].findall('QVAS_Contour')
        if len(conts):
            avail_slices.append(dicom_slicei)
    return avail_slices

def getContour(qvsroot,dicomslicei,conttype,dcmsz=720):
    qvasimg = qvsroot.findall('QVAS_Image')
    if dicomslicei - 1 > len(qvasimg):
        print('no slice', dicomslicei)
        return
    assert int(qvasimg[dicomslicei - 1].get('ImageName').split('I')[-1]) == dicomslicei
    conts = qvasimg[dicomslicei - 1].findall('QVAS_Contour')
    tconti = -1
    for conti in range(len(conts)):
        if conts[conti].find('ContourType').text == conttype:
            tconti = conti
            break
    if tconti == -1:
        print('no such contour', conttype)
        return
    pts = conts[tconti].find('Contour_Point').findall('Point')
    contours = []
    for pti in pts:
        contx = float(pti.get('x')) / 512 * dcmsz 
        conty = float(pti.get('y')) / 512 * dcmsz 
        #if current pt is different from last pt, add to contours
        if len(contours) == 0 or contours[-1][0] != contx or contours[-1][1] != conty:
            contours.append([contx, conty])
    return np.array(contours)    

def readOneDicom(path):
    slice1 = pydicom.read_file(path).pixel_array
    dcm_size = 720
    dcm_img = np.zeros((dcm_size, dcm_size))
    offset1 = dcm_size//2-slice1.shape[0]//2
    offset2 = dcm_size//2-slice1.shape[1]//2
    
    dcm_img[offset1:offset1+slice1.shape[0],
            offset2:offset2+slice1.shape[1]] = slice1
    return dcm_img   




path='./careIIChallenge/careIIChallenge'
outpath = './data/test'
cases = os.listdir(path)
count = 0
count2 = 0
for casei in cases[20:25]:
    #os.mkdir(outpath+"/"+casei)
    pi = casei.split('_')[1]
    dcm_img = readDicom(path+'/'+casei)   
    img2arti = {}
    img2lumen={}
    for arti in ['ICAL','ICAR','ECAL','ECAR']:
        cas_dir = path+'/'+casei+'/CASCADE-'+arti
        qvs_path = cas_dir+'/E'+pi+'S101_L.QVS'
        qvsroot = ET.parse(qvs_path).getroot()
        avail_slices = listContourSlices(qvsroot, dcm_img)
        
        for dicom_slicei in avail_slices:
            #self.filelist.append(self.path+'/'+casei+'/'+'E'+pi+'S101I%d.dcm'%dicom_slicei )
            cont = getContour(qvsroot,dicom_slicei,'Outer Wall')
            lumen = getContour(qvsroot,dicom_slicei,'Lumen')
            cont[:,1]-=280
            lumen[:,1]-=280
            if dicom_slicei in img2arti:
                img2arti[dicom_slicei].append(cont)
            else:
                img2arti[dicom_slicei] = []
                img2arti[dicom_slicei].append(cont)
                
            if dicom_slicei in img2lumen:
                img2lumen[dicom_slicei].append(lumen)
            else:
                img2lumen[dicom_slicei] = []
                img2lumen[dicom_slicei].append(lumen)
            #self.lumen_conts.append(self.getContour(qvsroot,dicom_slicei,'Lumen'))
            
    for k in range(dcm_img.shape[2]):       
        file = path+'/'+casei+'/'+'E'+pi+'S101I%d.dcm'%(k+1)
        x = readOneDicom(file)
        name = casei+'_E'+pi+'S101I%d'%(k+1)
        if k+1 in img2lumen:
            if len(img2lumen[k+1])==4:
                print("OK4")
            if len(img2lumen[k+1])==3:
                print("OK4")

        #conts = {}
        """
        if k+1 in img2lumen:
            x = x[280:280+160, :]

            outputImg = im.fromarray(x)
            outputImg = outputImg.convert('L')
            outputImg.save(outpath+"/"+name+".bmp")            
            conts["lumen"] = [  p.tolist()  for p in img2lumen[k+1]]
            conts["wall"] = [  p.tolist()  for p in img2arti[k+1]]
            with open(outpath+"/"+name+".json","w") as f:
                json.dump(conts, f)
            """
                
            
        

        
            