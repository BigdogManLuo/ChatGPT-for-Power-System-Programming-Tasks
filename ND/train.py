from models import divingModel
import torch
from torch import nn
from utilities import loadDataset,divingDataset,lossFunc
from torch import optim
import numpy as np
import datetime
from matplotlib import pyplot as plt


#定义模型
model=divingModel(nfeat=1,nhid=64,nclass=1,dropout=0.5,nNodes=1434,nOut=120,nlhid=256)  #nNodes和nOut需要调，nNodes约束总数+决策变量总数； nOut：决策变量数

#损失函数
lossfunc=nn.MSELoss()

#优化器
optimizer = optim.Adam(model.parameters(),lr=1e-4,weight_decay=5e-4)

#随机数种子
seed=42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

#训练数据上传到GPU
device=(torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
model=model.to(device=device)

#%%training loop
epoch_size=100
batch_size=10
group_size=2
#加载验证集
adjs,features,sols,objs=loadDataset(idx=-2)
valid_dataset=divingDataset(adjs,features,sols,objs)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                           batch_size=batch_size,
                                           drop_last = True ,      
                                           shuffle=True,
                                           )
Loss_train=[]
Loss_valid=[]
#训练循环                                              
for epoch in range(epoch_size): 
    model.train()
    for group in range(1,group_size):
        #加载训练数据
        adjs,features,sols,objs=loadDataset(idx=group)
        #构造训练集
        train_dataset=divingDataset(adjs,features,sols,objs)
        #构造迭代器
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   drop_last = True ,      
                                                   shuffle=True,
                                                   )
        loss_train=0
        for adj,feature,sol,obj in train_loader:
            #数据上传至GPU
            adj=adj.to(device=device)
            feature=feature.to(device=device)
            sol=sol.to(device=device)
            obj=obj.to(device=device)
            
            adj=torch.tensor(adj,dtype=torch.float32)
            
            #前向传播
            outputs=model(feature,adj)
            #计算损失
            loss=lossfunc(outputs,sol)
            #反向传播
            optimizer.zero_grad()
            loss.backward()
            #梯度下降
            optimizer.step()
            #记录损失
            loss_train+=loss.item()
        
        #记录每个group中训练集上的平均损失
        loss_train_mean=loss_train/len(train_loader)
        
        print('{} Epoch {} Group {}, trainLoss {} '.format(datetime.datetime.now(),epoch,group,loss_train_mean))    
    

    #验证
    model.eval()
    with torch.no_grad():
        loss_valid=0
        for adj,feature,sol,obj in valid_loader:
            #数据上传至GPU
            adj=adj.to(device=device)
            feature=feature.to(device=device)
            sol=sol.to(device=device)
            obj=obj.to(device=device)
            #修改为float类型
            adj=torch.tensor(adj,dtype=torch.float32)
            #前向传播
            outputs=model(feature,adj)
            #计算损失
            loss=lossfunc(outputs,sol)
            #记录损失
            loss_valid+=loss.item()
    
    loss_valid_mean=loss_valid/len(valid_loader)
    
    print('{} Epoch {} validLoss {} '.format(datetime.datetime.now(),epoch,loss_valid_mean))    
    
    #可视
    Loss_train.append(loss_train_mean)
    Loss_valid.append(loss_valid_mean)


#%% 可视化
plt.plot(Loss_train,label="train")
plt.plot(Loss_valid,label="valid")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.savefig("loss.png",dpi=400)
plt.show()
    
#保存模型
torch.save({'model': model.state_dict()}, 'model/diving_model.pth')

#%% 测试
import pickle
with open("data/samples/uc/test/adjs.pkl","rb") as f:
    adjs_test=pickle.load(f)
with open("data/samples/uc/test/features.pkl","rb") as f:
    features_test=pickle.load(f)
with open("data/samples/uc/test/sols.pkl","rb") as f:
    sols_test=pickle.load(f)
with open("data/samples/uc/test/objs.pkl","rb") as f:
    objs_test=pickle.load(f)

with torch.no_grad():
    model.eval() #不启用BatchNormalization 和 Dropout
    sol_pred=model(features_test,adjs_test.float())








