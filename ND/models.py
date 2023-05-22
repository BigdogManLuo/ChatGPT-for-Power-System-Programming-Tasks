import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid
from layers import GraphConvolution


class GCN(nn.Module):
    """
    图卷积网络：做两层图卷积，输出为节点的embedding
    """
    def __init__(self, nfeat, nhid, nclass, dropout):
        
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        
        #权重初始化
        for m in self.modules():
            if isinstance(m,GraphConvolution):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m,nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
        

    def forward(self, x, adj):
        x=x.unsqueeze(2)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        #return F.log_softmax(x, dim=1)
        return x


class MLP(nn.Module):
    """
    多层感知机：输入为节点的embedding向量,B x nNodes x 1;
    输出为  B x nNodes tensor
    """
    def __init__(self,nNodes,nOut,nlhid,dropout):
        super(MLP,self).__init__()
        self.hidden=nn.Linear(nNodes,nlhid)
        self.out=nn.Linear(nlhid,nOut)
        self.dropout = dropout
        #权重初始化
        for m in self.modules():
            if isinstance(m,nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
        
        
    def forward(self,x):
        x=x.squeeze(2)
        x=F.relu(self.hidden(x))
        x=F.dropout(x, self.dropout, training=self.training)
        return  self.out(x)
    
    
class divingModel(nn.Module):
    """
    图卷积网络与多层感知机的组合
    """
    def __init__(self,nfeat,nhid,nclass,dropout,nNodes,nOut,nlhid):
        super(divingModel,self).__init__()
        self.gcn=GCN(nfeat,nhid,nclass,dropout) #初始化图卷积网络
        self.mlp=MLP(nNodes,nOut,nlhid,dropout)   #初始化多层感知机

    def forward(self,x,adj):
        o1=self.gcn(x,adj)   
        o2=self.mlp(o1)      
        o3=sigmoid(o2)
        return o3
        
        
        
        
        
        
        
    