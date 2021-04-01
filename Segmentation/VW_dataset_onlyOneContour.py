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
        self.tx_vt = torchvision.transforms.Compose([
                        torchvision.transforms.ToPILImage(),
                        torchvision.transforms.CenterCrop([160,720]),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
                    ])
        self.tm_vt = torchvision.transforms.Compose([
                        torchvision.transforms.ToPILImage(),
                        torchvision.transforms.CenterCrop([160,720]),
                        torchvision.transforms.ToTensor()
                    ])
        self.tx = torchvision.transforms.Compose([
                        #torchvision.transforms.Resize((128,128)),                       
                        torchvision.transforms.ToPILImage(),
                        torchvision.transforms.CenterCrop([160,720]),
                        torchvision.transforms.RandomRotation((-10,10)),
                        torchvision.transforms.RandomHorizontalFlip(p=0.5),
                        torchvision.transforms.RandomVerticalFlip(p=0.5),
                        #torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
                    ])
        self.tm = torchvision.transforms.Compose([
                        #torchvision.transforms.Resize((128,128)),                      
                        torchvision.transforms.ToPILImage(),
                        torchvision.transforms.CenterCrop([160,720]),
                        torchvision.transforms.RandomRotation((-10,10)),
                        torchvision.transforms.RandomHorizontalFlip(p=0.5),
                        torchvision.transforms.RandomVerticalFlip(p=0.5),
                        #torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                        torchvision.transforms.ToTensor()
                        #torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
                    ])
        self.for_what = for_what
        self.wall_conts =[]
        #with open(path+file,'r') as f:
        #    self.filelist = f.readlines()
        cases = os.listdir(self.path)
        cur_cases=[]
        
        if for_what == 'train':
            cur_cases = cases[0:20]
        elif for_what == 'val':
            cur_cases = cases[19:20]
        elif for_what == 'test':
            cur_cases = cases[20:25]
            
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
                    
    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        file = self.filelist[idx]
        x, offset1, offset2=self.readOneDicom(file)
        #print(image.shape)
        x = x.reshape(1, x.shape[0], x.shape[1])#CxHxW 1 720 720     
        wall=self.wall_conts[idx]
        lumen=self.lumen_conts[idx]        
        # change lumen contour to mask
        mask = np.zeros((1,720,720), dtype=np.float32)
        mask = polygon2mask((720,720), wall)
        mask = np.transpose(mask)
        mask = mask.reshape(1, mask.shape[0], mask.shape[1])#CxHxW 1 720 720 
        x =x/255       
        x = torch.tensor(x, dtype=torch.float32)
        #wall = torch.tensor(wall, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        
        l = np.array(lumen)# real controid of lumen
        cl = np.array([np.mean(l[:,0]), np.mean(l[:,1])-280])
        cl = torch.tensor(cl, dtype=torch.float32)
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
        if self.for_what =='test' or self.for_what == 'val':
            x = self.tx_vt(x)
            mask = self.tm_vt(mask)
            return x, mask, cl
        else:
            seed=np.random.randint(0,2**31)
            random.seed(seed) 
            torch.manual_seed(seed)
            x = self.tx(x)  
            random.seed(seed) 
            torch.manual_seed(seed)
            mask = self.tm(mask)
            return x, mask, cl
    
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
        return dcm_img, np.array(sh)
    
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
    Dataset_val = VesselSet('val')
    Dataset_test = VesselSet('test')
    loader_train = torch.utils.data.DataLoader(dataset=Dataset_train,batch_size = 4, shuffle = False, num_workers=0)
    loader_val = torch.utils.data.DataLoader(dataset=Dataset_val,batch_size = 4, shuffle = False, num_workers=0)           
    loader_test = torch.utils.data.DataLoader(dataset=Dataset_test,batch_size = 4, shuffle = False, num_workers=0) 
    print(len(loader_test))
    x, m, _ = next(iter(loader_test))
    print(x.size())
    