import pickle
from torch.utils.data import Dataset
import torch
from torch import exp,log

def loadDataset(idx):
    if idx==-1: #Load test set
        filePath="data/samples/uc/test/"
    elif idx==-2: #Load valid set
        filePath="data/samples/uc/valid/"
    else: #Load train set
        filePath="data/samples/uc/train"+str(idx)+"/"
        
    adjs=pickle.load(open(filePath+"adjs.pkl","rb"))
    features=pickle.load(open(filePath+"features.pkl","rb"))
    sols=pickle.load(open(filePath+"sols.pkl","rb"))
    objs=pickle.load(open(filePath+"objs.pkl","rb"))

    return adjs,features,sols,objs

class divingDataset(Dataset):
    """
    divingDataset inherits Dataset, overloads __init__(), __getitem__(), __len__()
    Encapsulates a set of Tensor data into a Tensor dataset
    Can get the data of the dataset through index, and can get the size of the dataset through len
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
    Loss function
    Parameters
    ----------
    y : Neural network output batch_size x nVars
    sols : Set of feasible solutions batch_size x nSols x nVars
    objs : Objective function value corresponding to feasible solution batch_size x nSols

    Returns
    -------
    loss : Loss on the current batch
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
