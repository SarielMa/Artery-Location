# -*- coding: utf-8 -*-
import os 
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import torch
import torch.nn.functional as nnF
from torch.optim import Adam
from skimage.draw import polygon2mask
from Unet import Resnet18Unet, U_Net
#from u2net import U2NET, U2NETP
from tqdm import tqdm
import sys
from VW_dataset import VesselSet
import argparse
from imantics import Polygons, Mask
#from PCA_Aug import PCA_Aug_Dataloader
#%% deterministic is impossible, so let's do this
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
#%%
def getEuclidean(a, b):
    return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)
def distanceSum(ps1, ps2):#nx2 and mx2, all lists
    pairs = []
    pset1 = set()
    pset2 = set()
    ret = 0
    for i,p1 in enumerate(ps1):
        for j, p2 in enumerate(ps2):
            pairs.append([(i,j), getEuclidean(p1,p2)])
    pairs.sort(key = lambda x: x[1])
    
    for pai, distance in pairs:
        i,j = pai
        if i in pset1 or j in pset2:
            continue
        ret += distance
        pset1.add(i)
        pset2.add(j)
    return ret
#%%
def dice(Mp, M, reduction='none'):
    # NxKx128x128
    intersection = (Mp*M).sum(dim=(2,3))
    dice = (2*intersection) / (Mp.sum(dim=(2,3)) + M.sum(dim=(2,3))+1e-8)
    if reduction == 'mean':
        dice = dice.mean()
    elif reduction == 'sum':
        dice = dice.sum()
    return dice
#%%
def poly_disk(S):
    device=S.device
    S=S.detach().cpu().numpy()
    Mask = np.zeros((S.shape[0],1,720,720), dtype=np.float32)
    for i in range(S.shape[0]):
        Mask[i,0]=polygon2mask((720,720), S[i])
        Mask[i,0]=np.transpose(Mask[i,0])
    Mask = torch.tensor(Mask, dtype=torch.float32, device=device)
    return Mask
#%%
def dice_shape(Sp, S):
    S=S.view(S.shape[0], -1, 2)
    Sp=Sp.view(Sp.shape[0], -1, 2)
    M = poly_disk(S)
    Mp = poly_disk(Sp)
    score = dice(Mp, M)
    return score
#%%
def mrse_shape(Sp, S):
    S=S.view(S.shape[0], -1, 2)
    Sp=Sp.view(Sp.shape[0], -1, 2)
    error = ((Sp-S)**2).sum(dim=2).sqrt().mean(dim=1)
    return error
#%%
def update_lr(optimizer, new_lr):
    for g in optimizer.param_groups:
        g['lr']=new_lr
        print('new lr=', g['lr'])


#%%
def test(model, device, dataloader):
    model.eval()
    model.output='seg'
    
    dice_test=[]
    detected = 0
    distances = 0
    discovers = 0
    objs = 0
    with torch.no_grad():
        for batch_idx, (X, M, Cs) in enumerate(dataloader):

            X = X.to(torch.float32).to(device)  
            M = M.to(torch.float32).to(device)
            Mp=model(X) 
            
            Mp=Mp>0.5
            temp=dice(Mp, M)
            #if temp.item() >0.9:
            #    print(temp.item())
                
            dice_test.append(temp.detach().cpu().numpy()) 
            
            """
            cl = C.numpy() 
            Mp = Mp.cpu().numpy()   
            cps = []
            for i in range(Mp.shape[0]):
                polygons =Mask(Mp[i][0]).polygons()    
                cp = np.array([0.0,0.0])
                if(len(polygons.points)>0):
                    points = np.concatenate(polygons.points)
                    cp = np.array([np.mean(points[:,0]),np.mean(points[:,1])])
                    #dis = np.sqrt(np.sum((cp-cl[i])**2))
                    #mdis.append(dis)
                cps.append(cp)
                
            cps = np.array(cps)
            dis = np.sqrt(np.sum((cps-cl)**2, axis = 1))   
            if temp.item() >0.9:
                print("dis is ",dis) 
            """
            # because batch size is only 1
            #get ps2
            ps2 = []
            polygons =Mask(Mp[0][0].cpu().numpy()).polygons()
            ps = polygons.points
            for p in ps:
                ps2.append([np.mean(p[:,0]),np.mean(p[:,1])])
            #get ps1
            Cs = Cs.view(Cs.size()[1],Cs.size()[2]).cpu().numpy()
            ps1 = []
            temp = np.array([0.0,0.0])
            for c in Cs:     
                if not np.array_equal(c,temp):
                    ps1.append(c.tolist())
                    
                    
            if len(ps1)>1:
                print ("good")
            dis = distanceSum(ps1, ps2)
            
            distances += dis
            detected += min(len(ps1),len(ps2))
            discovers += len(ps2)
            objs += len(ps1)
            #see x and two mask
            """
            x = torch.squeeze(X).detach().cpu().numpy()
            m = torch.squeeze(M).detach().cpu().numpy()
            mp = torch.squeeze(Mp).detach().cpu().numpy()
            sp = torch.squeeze(Sp).detach().cpu().numpy()
            
            plt.figure(figsize=(10,10))
            plt.imshow(x)
            plt.plot(sp[:,0],sp[:,1],'ro',markersize=1)
            plt.show()
            plt.figure(figsize=(10,10))
            plt.imshow(m)
            plt.show()    
            plt.figure(figsize=(10,10))
            plt.imshow(mp)
            plt.show()
            """
            # distance in the centroid    
            """
            cl = C.numpy() 
            Mp = Mp.cpu().numpy()   
            cps = []
            for i in range(Mp.shape[0]):
                polygons =Mask(Mp[i][0]).polygons()
                a = np.array(polygons.points)
                if(len(a.shape)>1):
                    cp = np.array([np.mean(a[:,0,0]),np.mean(a[:,0,1])])
                else:
                    cp = np.array([0.0,0.0])
                cps.append(cp)
            cps = np.array(cps)
            dis = np.sqrt(np.sum((cps-cl)**2, axis = 1))          
            mdis.append(dis)   
            """

    #------------------
    dice_test=np.concatenate(dice_test)
    mdis=distances/detected
    return mdis, np.mean(dice_test),discovers,objs, detected

#%%
def plot_history(history):
    fig, ax = plt.subplots(2, 3, constrained_layout=True)
    ax[0,0].plot(history['loss1_train'], '-b', label='loss1_train')
    ax[0,0].plot(history['loss2_train'], '-c', label='loss2_train')
    ax[0,0].plot(history['loss3_train'], '-g', label='loss3_train') 
    ax[0,0].plot(history['loss4_train'], '-g', label='loss4_train')
    ax[0,0].grid(True)
    ax[0,0].legend()

    #==============================euclidean distance between centroid=============================
    ax[0,1].plot(history['mdis_val'],   '-r', label='mDis_val')
    ax[0,1].set_ylim(0, 100)
    #ax[0,1].set_yticks(np.linspace(0, 3, 7))
    ax[0,1].grid(True)
    ax[0,1].legend() 

    ax[0,2].plot(history['mdis_test'], '-r', label='mDis_test')
    ax[0,2].set_ylim(0, 100)
    #ax[0,2].set_yticks(np.linspace(0, 3, 7))
    ax[0,2].grid(True)
    ax[0,2].legend() 
    #===============================dice index======================================================
    ax[1,0].plot(history['dice_train'], '-c', label='dice_train')
    ax[1,0].set_ylim(0, 1)
    #ax[1,0].set_yticks(np.linspace(0.84, 1, 9))
    ax[1,0].grid(True)
    ax[1,0].legend()


    ax[1,1].plot(history['dice_val'], '-c', label='dice_val')
    ax[1,1].set_ylim(0, 1)
    #ax[1,1].set_yticks(np.linspace(0.84, 1, 9))
    ax[1,1].grid(True)
    ax[1,1].legend()


    ax[1,2].plot(history['dice_test'], '-c', label='dice_test')
    ax[1,2].set_ylim(0, 1)
    #ax[1,2].set_yticks(np.linspace(0.84, 1, 9))
    ax[1,2].grid(True)
    ax[1,2].legend()

    return fig, ax  
#%%
def save_checkpoint(filename, model, optimizer, history, epoch):
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history':history},
               filename)
    print('saved:', filename)
#%%
def load_checkpoint(filename, model, optimizer, history):
    checkpoint=torch.load(filename, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  
    if history is not None:
        if 'history' in checkpoint.keys():
            history.update(checkpoint['history'])
        else:
            history.update(checkpoint['result'])
    print('loaded:', filename)
#%%
if __name__ == '__main__':
    #%%
    parser = argparse.ArgumentParser(description='Input Parameters:')
    parser.add_argument('--net_name', default='Original_Unet', type=str)
    parser.add_argument('--epoch_start', default=100, type=int)
    parser.add_argument('--epoch_end', default=100, type=int)
    parser.add_argument('--cuda_id', default=1, type=int)
    parser.add_argument('--path', default='./careIIChallenge/careIIChallenge', type=str)
    parser.add_argument('--path_aug', default='', type=str)
    arg = parser.parse_args()
    print(arg)
    device = torch.device("cuda:"+str(arg.cuda_id) if torch.cuda.is_available() else "cpu")
    #%%
    #Dataset_train = VesselSet( 'train')
    Dataset_test = VesselSet( 'test')
    #loader_train = torch.utils.data.DataLoader(dataset=Dataset_train,batch_size = 32, shuffle = True, num_workers=0)
    loader_test = torch.utils.data.DataLoader(dataset=Dataset_test,batch_size = 1, shuffle = False, num_workers=0)
    #%% validation using pca_aug data on training set
    #loader_val=PCA_Aug_Dataloader(n_epochs=1, n_batches=100, batch_size=64, device=device, shuffle = False,
    #                              filename=arg.path_aug+'pca_aug_P30b100n64')
    #%%
    filename='result/'+arg.net_name+'_VW_seg'
    print('save to', filename)
    #%%

    #model = Resnet18Unet(1).to(device)
    model = U_Net().to(device)
    #elif arg.net_name =='U2NET':
    #    model = U2NET(352,1).to(device)
    #elif arg.net_name =='U2NETP':
    #    model = U2NETP(352,1).to(device)
    optimizer = Adam(model.parameters(),lr = 0.001)
    history={'loss1_train':[], 'loss2_train':[], 'loss3_train':[],'loss4_train':[],
             'mdis_train':[], 'dice_train':[],
             'mdis_val':[], 'dice_val':[],
             'mdis_test':[],  'dice_test':[]}
    #%%
    #print(torch.cuda.memory_summary(device=device, abbreviated=True))
    #%% load model state and optimizer state if necessary
    epoch_save=arg.epoch_start-1
    if epoch_save>=0:
        load_checkpoint(filename+'_epoch'+str(epoch_save)+'.pt', model, optimizer, history)
        
    mdis_test, dice_test, discovers, objs, detected = test(model, device, loader_test)    
    

    print("val mean distance: ", mdis_test)
    print("val dice index: ", dice_test)
    print("discovers: ", discovers)
    print("objects: ", objs)
    print ("detected: ",detected)

    #history['mdis_test'].append(mdis_test)
    #history['dice_test'].append(dice_test) 

    #fig1, ax1 = plot_history(history) 
    #fig1.savefig(filename+'_epoch'+str(epoch_save)+'_history_test.png')
    #plt.close(fig1)           


        