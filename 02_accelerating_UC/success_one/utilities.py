import pickle
from torch.utils.data import Dataset
import torch
from torch import exp,log

def loadDataset(idx):
    if idx==-1: #加载测试集
        filePath="data/samples/uc/test/"
    elif idx==-2: #加载验证集
        filePath="data/samples/uc/valid/"
    else: #加载训练集
        filePath="data/samples/uc/train"+str(idx)+"/"
        
    adjs=pickle.load(open(filePath+"adjs.pkl","rb"))
    features=pickle.load(open(filePath+"features.pkl","rb"))
    sols=pickle.load(open(filePath+"sols.pkl","rb"))
    objs=pickle.load(open(filePath+"objs.pkl","rb"))
    '''
    if idx==-1: #如果是加载测试集数据
        c=pickle.load(open(filePath+"c.pkl","rb"))
        return adjs,features,sols,objs,c
    else:  #如果是加载训练集或验证集数据
        return adjs,features,sols,objs
    '''
    return adjs,features,sols,objs

class divingDataset(Dataset):
    """
    divingDataset继承Dataset, 重载了__init__(), __getitem__(), __len__()
    实现将一组Tensor数据对其封装成Tensor数据集
    能够通过index得到数据集的数据，能够通过len，得到数据集大小
    """
    def __init__(self,adjs,features,sols,objs):
        self.adjs = adjs
        self.features=features
        self.sols=sols
        self.objs=objs
        
    def __getitem__(self,index):
        return self.adjs[index], self.features[index],self.sols[index],self.objs[index]

    def __len__(self):
        return self.adjs.size(0)


    
def lossFunc(y,sols,objs):
    """
    损失函数
    Parameters
    ----------
    y : 神经网络输出 batch_size x nVars
    sols : 可行解集合 batch_size x nSols x nVars
    objs : 可行解对应的目标函数值 batch_size x nSols

    Returns
    -------
    loss : 在当前batch上的损失
    """
    objs=objs/100000
    eObjs=exp(-objs)
    den=eObjs.sum(axis=1)
    den=den.unsqueeze(1)
    w=eObjs/den
    y=y.unsqueeze(1)
    p=y*sols+(1-y)*(1-sols)
    p=log(p+1e-45)
    P=p.sum(axis=2)
    loss=-(w*P).sum()
    return loss
    
'''
def collate_func(batch):
    """
    在默认的dataloader里面会将dataset中的数据stack成tensor，而因为不同MIP问题的可行解个数可能不一样，
    因此数据集中的sols和objs的大小可能在每个batch中不一样，需要使得dataloader返回的sols和objs为list对象
    """
    
    adjs=[]
    features=[]
    sols=[]
    objs=[]
    
    for item in batch:
        # 因为token_list是一个变长的数据，所以需要用一个list来装这个batch的token_list
        adj=item[0]
        feature=item[1]
        sol=item[2]
        obj=item[3]
        
        adjs.append(adj)
        features.append(feature)
        sols.append(sol)
        objs.append(obj)
    
    adjs=torch.stack(adjs)
    features=torch.stack(features)
    
    
    return adjs,features,sols,objs
'''

#%% 单个batch的损失函数测试
"""
import torch
y1=torch.tensor([0.8,0.2,0.2])
y2=torch.tensor([0.2,0.8,0.2])
y3=torch.tensor([0.2,0.3,0.8])
sol=torch.tensor([[1,0,0],[0,1,0],[0,0,1]])
w=torch.tensor([0.09,0.245,0.665])
def lossFunc_test(y,sol):
    p=y*sol+(1-y)*(1-sol)
    p=log(p)
    P=p.sum(axis=1)
    loss=-(P*w).sum()
    return loss

print("y1 loss: {} ".format(lossFunc_test(y1,sol)))
print("y2 loss: {} ".format(lossFunc_test(y2,sol)))
print("y3 loss: {} ".format(lossFunc_test(y3,sol)))
"""

"""
#%% batch_size=2的损失函数测试
sols=[]
sol1=torch.tensor([[1,0,1],[0,1,0],[0,0,1],[1,1,1]])
sol2=torch.tensor([[1,0,0],[0,1,0],[0,0,1],[1,1,1]])
sols.append(sol1)
sols.append(sol2)
sols=torch.stack(sols)
objs=torch.tensor([[6,2,1,8],[3,2,1,6]])

y1=torch.tensor([[0.8,0.2,0.8],[0.8,0.2,0.2]])
y2=torch.tensor([[0.2,0.8,0.2],[0.2,0.8,0.2]])
y3=torch.tensor([[0.2,0.2,0.8],[0.2,0.2,0.8]])
y4=torch.tensor([[0.7,0.6,0.8],[0.7,0.8,0.8]])

print("y1 loss: {} ".format(lossFunc(y1,sols,objs)))
print("y2 loss: {} ".format(lossFunc(y2,sols,objs)))
print("y3 loss: {} ".format(lossFunc(y3,sols,objs)))
print("y4 loss: {} ".format(lossFunc(y4,sols,objs)))
"""

'''
#%% 在setcover数据集上的损失函数测试
import matplotlib.pyplot as plt
adjs,features,sols,objs=loadDataset(idx=1)
B=adjs.shape[0]
nVars=120

#让y_pred模仿不同组的可行解
y_pred_all=[]

for k in range(10):
    y_pred=torch.ones(B,nVars)
    for i in range(B):
        for j in range(nVars):
            if sols[i,k,j]==1:
                y_pred[i,j]=0.9
            elif sols[i,k,j]==0:
                y_pred[i,j]=0.1
    y_pred_all.append(y_pred)

y_rand=torch.rand((10,120))

loss_all=[]    
for k in range(10):   
    loss=lossFunc(y_pred_all[k],sols,objs)
    loss_all.append(loss)

plt.plot(loss_all)
'''


