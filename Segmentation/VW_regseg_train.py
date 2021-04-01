# -*- coding: utf-8 -*-
import os 
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import torch
import torch.nn.functional as nnF
from torch.optim import Adam
from skimage.draw import polygon2mask
from Unet import Resnet18Unet
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
# M = M*1*H* W
def Must2Polygon(M): 
    pass
#%%
def update_lr(optimizer, new_lr):
    for g in optimizer.param_groups:
        g['lr']=new_lr
        print('new lr=', g['lr'])
#%%
def train(model, device, optimizer, dataloader, epoch):
    model.train()
    loss1_train=0
    loss2_train=0
    loss3_train=0
    loss4_train=0
    for batch_idx, (X, M, _) in enumerate(dataloader):
        # M is from 0 to 1
        X, M = X.to(device), M.to(device)
        #M=poly_disk(S)
        Mp=model(X)
        loss_ce=nnF.binary_cross_entropy_with_logits(Mp, M)               
        
        Mp=torch.sigmoid(Mp)
        loss_dice=1-dice(Mp, M, 'mean')
        #--------------------------------------
        #Sp=Sp.view(Sp.shape[0], -1, 2)
        #loss_mae=(S-Sp).abs().mean()    
        #--------------------------------------  
        #CSp = torch.mean(Sp, dim = 1)
        #loss_dis = torch.sqrt(torch.sum((CSp-C)**2, dim = 1)).mean()
        #--------------------------------------
        loss=loss_ce + loss_dice*3
        #loss= loss_mae + loss_dis
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss1_train+=loss_ce.item()
        loss2_train+=loss_dice.item()
        #loss3_train+=loss_mae.item()
        #loss4_train+=loss_dis.item()
    loss1_train/=len(dataloader)
    loss2_train/=len(dataloader)
    loss3_train/=len(dataloader)
    loss4_train/=len(dataloader)
    print("training dice is :", 1-loss2_train)
    print ("training bce loss is : ", loss1_train)
    return loss1_train, loss2_train, loss3_train, loss4_train

#%%
def test(model, device, dataloader):
    model.eval()
    model.output='seg'
    mdis = []
    dice_test=[]
    
    with torch.no_grad():
        for batch_idx, (X, M, C) in enumerate(dataloader):
            #M=poly_disk(S).to(device)
            X = X.to(torch.float32).to(device)  
            M = M.to(torch.float32).to(device)
            Mp=model(X) 
            #Sp=Sp.view(Sp.shape[0], -1, 2)
            #Mp=poly_disk(Sp).to(device)
            # first dice index
            Mp=Mp>0.5
            temp=dice(Mp, M)
            dice_test.append(temp.detach().cpu().numpy()) 
            
            # distance in the centroid    
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

                cps.append(cp)
            cps = np.array(cps)
            dis = np.sqrt(np.sum((cps-cl)**2, axis = 1))          
            mdis.append(dis)  
            """
            """
            Sp = Sp.cpu()
            #Sp=Sp.view(Sp.shape[0], -1, 2)
            CSp = torch.mean(Sp, dim = 1)
            dis = torch.sqrt(torch.sum((CSp-C)**2, dim = 1))
            mdis.append(dis)
            """
    #------------------
    dice_test=np.concatenate(dice_test)
    #mdis=np.concatenate(mdis)
    return 0, np.mean(dice_test)

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
    parser.add_argument('--net_name', default='Resnet18Unet', type=str)
    parser.add_argument('--epoch_start', default=80, type=int)
    parser.add_argument('--epoch_end', default=200, type=int)
    parser.add_argument('--cuda_id', default=0, type=int)
    parser.add_argument('--path', default='./careIIChallenge/careIIChallenge', type=str)
    parser.add_argument('--path_aug', default='', type=str)
    arg = parser.parse_args()
    print(arg)
    device = torch.device("cuda:"+str(arg.cuda_id) if torch.cuda.is_available() else "cpu")
    #%%
    Dataset_train = VesselSet( 'train')
    Dataset_val = VesselSet( 'test')
    loader_train = torch.utils.data.DataLoader(dataset=Dataset_train,batch_size = 64, shuffle = True, num_workers=0)
    loader_val = torch.utils.data.DataLoader(dataset=Dataset_val,batch_size = 64, shuffle = False, num_workers=0)
    #%% validation using pca_aug data on training set
    #loader_val=PCA_Aug_Dataloader(n_epochs=1, n_batches=100, batch_size=64, device=device, shuffle = False,
    #                              filename=arg.path_aug+'pca_aug_P30b100n64')
    #%%
    filename='result/'+arg.net_name+'_VW_seg'
    print('save to', filename)
    #%%

    model = Resnet18Unet(1).to(device)
    #model = Resnet18Unet(36,1).to(device)

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
    #%%
    for epoch in tqdm(range(epoch_save+1, arg.epoch_end), initial=epoch_save+1, total=arg.epoch_end):
        if epoch == 200:
            update_lr(optimizer, 0.0001)
        elif epoch == 400:
            update_lr(optimizer, 0.00001)
        loss_train = train(model, device, optimizer, loader_train, epoch)
        #print(torch.cuda.memory_summary(device=device, abbreviated=True))
        #mrse_train, dice1_train, dice2_train = test(model, device, loader_train)
        #mdis_train, dice_train = test(model, device, loader_train)
        mdis_val, dice_val = test(model, device, loader_val)
        print("val mean distance: ", mdis_val)
        print("val dice index: ", dice_val)
        history['loss1_train'].append(loss_train[0])
        history['loss2_train'].append(loss_train[1])
        history['loss3_train'].append(loss_train[2])
        history['loss4_train'].append(loss_train[3])
        #history['mdis_train'].append(mdis_train)
        #history['dice_train'].append(dice_train)
        history['mdis_val'].append(mdis_val)
        history['dice_val'].append(dice_val) 
        #history['mdis_test'].append(mdis_val.mean())
        #history['dice_test'].append(dice_val.mean()) 
        #------- show result ----------------------
        #display.clear_output(wait=False)
        fig1, ax1 = plot_history(history)            
        #display.display(fig1)        
        fig2, ax2 = plt.subplots()
        ax2.hist(mdis_val, bins=50, range=(0,10))
        ax2.set_xlim(0, 10)
        #display.display(fig2)
        #----------save----------------------------
        if (epoch+1)%5 == 0:
            save_checkpoint(filename+'_epoch'+str(epoch)+'.pt', model, optimizer, history, epoch)
            fig1.savefig(filename+'_epoch'+str(epoch)+'_history.png')
            fig2.savefig(filename+'_epoch'+str(epoch)+'_mdis_test.png')
        epoch_save=epoch
        plt.close(fig1)
        plt.close(fig2)
