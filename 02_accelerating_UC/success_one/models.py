import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid
from layers import GraphConvolution


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support) + self.bias
        return output



class GCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout):
        
        super(GCN, self).__init__()
        self.gc1 = GCNLayer(nfeat, nhid)
        self.gc2 = GCNLayer(nhid, nclass)
        self.dropout = dropout
        
        #Initialize weights
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

    def __init__(self,nNodes,nOut,nlhid,dropout):
        super(MLP,self).__init__()
        self.hidden=nn.Linear(nNodes,nlhid)
        self.out=nn.Linear(nlhid,nOut)
        self.dropout = dropout
        #Initialize weights
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

    def __init__(self,nfeat,nhid,nclass,dropout,nNodes,nOut,nlhid):
        super(divingModel,self).__init__()
        self.gcn=GCN(nfeat,nhid,nclass,dropout)
        self.mlp=MLP(nNodes,nOut,nlhid,dropout)   

    def forward(self,x,adj):
        o1=self.gcn(x,adj)   
        o2=self.mlp(o1)      
        o3=sigmoid(o2)
        return o3
        
        
        
        
        
        
        
    