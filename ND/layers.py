import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.autograd import Function

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight) #修改为高维矩阵乘法
        output = torch.matmul(adj, support)#修改为高维矩阵乘法
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



'''
class Mapping(Function):
    
    """
    输出映射，将0-1范围的输出映射到(0,0.05)U(0.95,1)范围，提高数值稳定性
    """
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)         # 将输入保存起来，在backward时使用
        output=torch.zeros_like(input_,device=input_.device)
        idx1=torch.where(input_<0.5)
        idx2=torch.where(input_>=0.5)
        output[idx1]=input_[idx1]*0.1
        output[idx2]=input_[idx2]*0.1+0.9
        return output
    
    @staticmethod
    def backward(ctx,grad_output):
        input_,=ctx.saved_tensors
        output=torch.ones(input_.shape,device=input_.device)
        return grad_output*output*0.1


"""    
#%% 对Mapping 自定义层做梯度检验

x=torch.rand(3,100,requires_grad=True,dtype=torch.float32)
test=torch.autograd.gradcheck(Mapping.apply, (x,), eps=1e-3)

#检查映射
x=torch.linspace(0,1,100)
mapping=Mapping.apply
y=mapping(x)

import matplotlib.pyplot as plt
plt.plot(x,y)
plt.yticks([0,0.05,0.95,1])
plt.grid()
"""
'''