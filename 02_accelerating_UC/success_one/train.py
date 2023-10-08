from models import divingModel
import torch
from torch import nn
from utilities import loadDataset,divingDataset,lossFunc
from torch import optim
import numpy as np
import datetime
from matplotlib import pyplot as plt


#Define model
model=divingModel(nfeat=1,nhid=64,nclass=1,dropout=0.5,nNodes=2844,nOut=240,nlhid=256)  
lossfunc=nn.MSELoss()

#Optimizer
optimizer = optim.Adam(model.parameters(),lr=5e-3,weight_decay=5e-4)

#Random seed
seed=42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

#Upload to GPU
device=(torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
model=model.to(device=device)

#%%training loop
epoch_size=100
batch_size=20
group_size=4
#Load validation dataset
adjs,features,sols,objs=loadDataset(idx=-2)
valid_dataset=divingDataset(adjs,features,sols,objs)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                           batch_size=batch_size,
                                           drop_last = True ,      
                                           shuffle=True,
                                           )
Loss_train=[]
Loss_valid=[]
#Training loop                                            
for epoch in range(epoch_size): 
    model.train()
    for group in range(1,group_size+1):
        #Load training dataset
        adjs,features,sols,objs=loadDataset(idx=group)
        #Make dataset
        train_dataset=divingDataset(adjs,features,sols,objs)
        #Define dataloader
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   drop_last = True ,      
                                                   shuffle=True,
                                                   )
        loss_train=0
        for adj,feature,sol,obj in train_loader:
            #Upload to GPU
            adj=adj.to(device=device)
            feature=feature.to(device=device)
            sol=sol.to(device=device)
            obj=obj.to(device=device)
            
            adj=torch.tensor(adj,dtype=torch.float32)
            
            #Forward Pass
            outputs=model(feature,adj)
            #Loss
            loss=lossfunc(outputs,sol)
            #Backward Pass
            optimizer.zero_grad()
            loss.backward()
            #Gradient Descent
            optimizer.step()
            #Record loss
            loss_train+=loss.item()
        
        #Record mean loss
        loss_train_mean=loss_train/len(train_loader)
        
        print('{} Epoch {} Group {}, trainLoss {} '.format(datetime.datetime.now(),epoch,group,loss_train_mean))    
    

    #Validation
    model.eval()
    with torch.no_grad():
        loss_valid=0
        for adj,feature,sol,obj in valid_loader:
            #Upload to GPU
            adj=adj.to(device=device)
            feature=feature.to(device=device)
            sol=sol.to(device=device)
            obj=obj.to(device=device)
            adj=torch.tensor(adj,dtype=torch.float32)
            #Forward Pass
            outputs=model(feature,adj)
            #Loss
            loss=lossfunc(outputs,sol)
            #Record loss
            loss_valid+=loss.item()
    
    loss_valid_mean=loss_valid/len(valid_loader)
    
    print('{} Epoch {} validLoss {} '.format(datetime.datetime.now(),epoch,loss_valid_mean))    
    
    #Visualize
    Loss_train.append(loss_train_mean)
    Loss_valid.append(loss_valid_mean)

#Save model
torch.save({'model': model.state_dict()}, 'model/diving_model.pth')


#%% Visualize
import matplotlib
matplotlib.rcParams['font.family']='Arial'     
matplotlib.rcParams['font.sans-serif'] = ['Arial']

plt.plot(Loss_train,label="train",color="#4a69bd")
plt.plot(Loss_valid,label="valid",color="#b71540")
plt.legend()
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.grid()


plt.savefig("loss.png",dpi=666)
