# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 19:38:17 2021

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
#path = /careIIChallenge/careIIChallenge
class VesselSet(torch.utils.data.Dataset):
    for_what=''
    def __init__(self, for_what):
        self.path='./careIIChallenge/careIIChallenge'
        self.filelist=[]
        self.lumen_conts=[]

        self.for_what = for_what
        
        cases = os.listdir(self.path)
        cur_cases=[]
        
        if for_what == 'train':
            cur_cases = cases[0:20]
        elif for_what == 'val':
            cur_cases = cases[19:20]
        elif for_what == 'test':
            cur_cases = cases[20:25]
        """            
        shapes = []
        for casei in cur_cases:
            pi = casei.split('_')[1]
            dcm_img,shape = self.readDicom(self.path+'/'+casei)   
            shapes.append(shape)
            for arti in ['ICAL','ICAR','ECAL','ECAR']:
                cas_dir = self.path+'/'+casei+'/CASCADE-'+arti
                qvs_path = cas_dir+'/E'+pi+'S101_L.QVS'
                qvsroot = ET.parse(qvs_path).getroot()
                avail_slices = self.listContourSlices(qvsroot, dcm_img)
                for dicom_slicei in avail_slices:
                    self.filelist.append(self.path+'/'+casei+'/'+'E'+pi+'S101I%d.dcm'%dicom_slicei )
                    self.lumen_conts.append(self.getContour(qvsroot,dicom_slicei,'Lumen'))
                    self.wall_conts.append(self.getContour(qvsroot,dicom_slicei,'Outer Wall'))
        shapes = np.concatenate(shapes)
        print ("OK")
        """
        #self.wall_conts =[]
        self.img2arti = {}
        self.img2lumen={}
        for casei in cur_cases:
            pi = casei.split('_')[1]
            dcm_img = self.readDicom(self.path+'/'+casei)   
            for arti in ['ICAL','ICAR','ECAL','ECAR']:
                cas_dir = self.path+'/'+casei+'/CASCADE-'+arti
                qvs_path = cas_dir+'/E'+pi+'S101_L.QVS'
                qvsroot = ET.parse(qvs_path).getroot()
                avail_slices = self.listContourSlices(qvsroot, dcm_img)
                
                for dicom_slicei in avail_slices:
                    lumen = self.getContour(qvsroot,dicom_slicei,'Lumen')
                    cont = self.getContour(qvsroot,dicom_slicei,'Outer Wall')
                    cur = self.path+'/'+casei+'/'+'E'+pi+'S101I%d.dcm'%dicom_slicei
                    if cur in self.img2arti:
                        self.img2arti[cur].append(cont)
                    else:
                        self.img2arti[cur] = []
                        self.img2arti[cur].append(cont)
                        
                    if cur in self.img2lumen:
                        self.img2lumen[cur].append(lumen)
                    else:
                        self.img2lumen[cur] = []
                        self.img2lumen[cur].append(lumen)
            """        
            for k in range(dcm_img.shape[2]):
                file = self.path+'/'+casei+'/'+'E'+pi+'S101I%d.dcm'%(k+1)
                x,_,_ =  self.readOneDicom(file)
                if k+1 in img2arti:
                    count2 +=1
                    ms = img2arti[k+1]
                    #plt.imshow(x)
                    #plt.plot(ms[0][:,0],ms[0][:,1],'ro',markersize=1)
                    #plt.show()
                    if len(ms)>1:
                        print("stop!!")
            """
            self.files = list(self.img2arti.keys())
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        x, _, _,=self.readOneDicom(file)
        #print(image.shape)
        #x = x.reshape(1, x.shape[0], x.shape[1])#CxHxW 1 160 720     
        #wall=self.wall_conts[idx]
        #lumen=self.lumen_conts[idx]   
        walls = self.img2arti[file]
        """
        if(len(walls) ==0 or len(walls) >1):
            print("stop")
        """
        # change lumen contour to mask


        x =x/255       
        x = torch.tensor(x, dtype=torch.float32)
        #wall = torch.tensor(wall, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        
        l = wall# real controid of wall

        
        """
        plt.figure(figsize=(10,10))
        plt.imshow(x[0])
        #plt.plot(lumen[:,0],lumen[:,1],'ro',markersize=1)
        plt.show()
        plt.figure(figsize=(10,10))
        plt.imshow(mask[0])
        plt.plot(lumen[:,0],lumen[:,1],'ro',markersize=1)
        plt.show()
        """
        return x, 
    
    def readOneDicom(self, path):
        slice1 = pydicom.read_file(path).pixel_array
        dcm_size = 720
        dcm_img = np.zeros((dcm_size, dcm_size))
        offset1 = dcm_size//2-slice1.shape[0]//2
        offset2 = dcm_size//2-slice1.shape[1]//2
        
        dcm_img[offset1:offset1+slice1.shape[0],
                offset2:offset2+slice1.shape[1]] = slice1
        return dcm_img, offset1, offset2
    
    def readDicom(self,path):
        pi = os.path.basename(path).split('_')[1]
        dcm_size = len(glob.glob(path+'/*.dcm'))
        dcms = [path+'/E'+pi+'S101I%d.dcm'%dicom_slicei for dicom_slicei in range(1,dcm_size+1)]
       #dcm_f = pydicom.read_file(dcms[0]).pixel_array
        dcm_size = 720
        dcm_img = np.zeros((dcm_size,dcm_size,len(dcms)))
        sh = []
        for dcmi in range(len(dcms)):
            cdcm = pydicom.read_file(dcms[dcmi]).pixel_array
            dcm_img[dcm_size//2-cdcm.shape[0]//2:dcm_size//2+cdcm.shape[0]//2,
                    dcm_size//2-cdcm.shape[1]//2:dcm_size//2+cdcm.shape[1]//2,dcmi] = cdcm
            
            sh.append([cdcm.shape[0], cdcm.shape[1]])
        return dcm_img
    
    def listContourSlices(self,qvsroot, dcm_img):
        avail_slices = []
        qvasimg = qvsroot.findall('QVAS_Image')
        for dicom_slicei in range(dcm_img.shape[2]):
            conts = qvasimg[dicom_slicei - 1].findall('QVAS_Contour')
            if len(conts):
                avail_slices.append(dicom_slicei)
        return avail_slices
    
    def getContour(self,qvsroot,dicomslicei,conttype,dcmsz=720):
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

if __name__=='__main__':
    Dataset_train = VesselSet('train')
    #Dataset_val = VesselSet('val')
    Dataset_test = VesselSet('test')
    loader_train = torch.utils.data.DataLoader(dataset=Dataset_train,batch_size =1 , shuffle = False, num_workers=0)
    #loader_val = torch.utils.data.DataLoader(dataset=Dataset_val,batch_size = 4, shuffle = False, num_workers=0)           
    loader_test = torch.utils.data.DataLoader(dataset=Dataset_test,batch_size = 1, shuffle = False, num_workers=0) 
    #print(len(loader_test))
    #x, m, _ = next(iter(loader_test))
    #print(x.size())
    
    