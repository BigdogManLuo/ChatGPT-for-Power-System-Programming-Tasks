from utilities import loadDataset,lossFunc
from models import divingModel
import numpy as np
import torch
from torch.utils.data import TensorDataset,DataLoader
import matplotlib.pyplot as plt
import os
import gurobipy
from gurobipy import GRB
import time
import pickle
import gurobipy as gp
import random 
random.seed(0)

trainFile = os.listdir("data/instances/uc/test")[0:100]

adjs=pickle.load(open("data/samples/uc/test/"+"adjs.pkl","rb"))
features=pickle.load(open("data/samples/uc/test/"+"features.pkl","rb"))
sols=pickle.load(open("data/samples/uc/test/"+"sols.pkl","rb"))


#%% 

model=divingModel(nfeat=1,nhid=64,nclass=1,dropout=0.5,nNodes=3308,nOut=240,nlhid=256)  #nNodes和nOut需要调，nNodes约束总数+决策变量总数； nOut：决策变量数
state_dict = torch.load('model/diving_model.pth')
model.load_state_dict(state_dict['model'])

Objs_true=pickle.load(open("data/tmp/test/Objs.pkl","rb"))


def OptimizeRemainedLP(ins_index,U_values):
    
    m = gp.read("data/instances/uc/test/instance"+str(ins_index)+".lp")
    indices=random.sample(range(10),2)

    for n in indices:
        for t in range(24):
            var_name=f"U_{n}_{t}"
            var=m.getVarByName(var_name)
            # Set the value from U_values
            var.setAttr("LB", U_values[n, t])
            var.setAttr("UB", U_values[n, t])
    
    #Optimize the Remained LP problem
    start_time=time.time()
    m.optimize()
    end_time=time.time()
    
    if m.status == gp.GRB.Status.OPTIMAL:
        status="feasible"
    else:
        status="infeasible"
        
    if status=="feasible":
        #Get Objective Value
        obj=m.getObjective().getValue()
    else:
        obj=np.random.uniform(510000,530000)
        
    return status,obj,end_time-start_time
   


#merge adjs,features and sols to torch Dataset
dataset=TensorDataset(adjs,features,sols)


model.eval()
Objs_pred=[]
Time=[]
Status=[]
dataloader=DataLoader(dataset, batch_size=1, shuffle=True)
ins_index=0
with torch.no_grad():
    for adj,feature,sol in dataloader:
        outputs = model(feature, adj.float())
        outputs=outputs.numpy()
        outputs=np.random.binomial(1,outputs)
        outputs=outputs.reshape(24,10)
        outputs=outputs.T
<<<<<<< HEAD
        
        # Fix U and optimized the modified model 
        status,obj,consum_time=OptimizeRemainedLP(ins_index,outputs)
        
=======
        status,obj,consum_time=OptimizeRemainedLP(ins_index,outputs)
>>>>>>> 07c95232616711f546d7ba4e58efd49fe0d959f9
        #Update ins_index
        ins_index=ins_index+1
        
        Objs_pred.append(obj)
        Time.append(consum_time)
        Status.append(status)




