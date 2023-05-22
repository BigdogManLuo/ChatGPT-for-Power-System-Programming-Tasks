import networkx as nx
import gurobipy
from gurobipy import GRB
import numpy as np
import os
import torch
import pickle
import multiprocessing

def mip_to_bipartite(lp_file_path):
    
    # Read the lp file using Gurobi
    model = gurobipy.read(lp_file_path)

    # Create a new bipartite graph
    B = nx.Graph()

    # Add nodes with the variable names and constraint names
    for var in model.getVars():
        # For variable nodes, we store the objective coefficient as a node attribute
        B.add_node(var.VarName, bipartite=0, obj_coeff=var.obj)  

    for constr in model.getConstrs():
        # For constraint nodes, we store the right-hand side value as a node attribute
        B.add_node(constr.ConstrName, bipartite=1, rhs=constr.RHS) 

        # For each variable in the constraint, add an edge between the
        # constraint node and the variable node
        for var in model.getVars():
            B.add_edge(var.VarName, constr.ConstrName)

    # Get the adjacency matrix (as a SciPy sparse matrix)
    adj = nx.adjacency_matrix(B)
    
    # Get the right-hand side values (b) of the constraints
    b = np.array([constr.RHS for constr in model.getConstrs()])

    # Get the objective function coefficients (c)
    c = np.array([var.Obj for var in model.getVars()])

    feature = np.concatenate((b, c), axis=0)
    
    return adj,feature


def solve_mip(lp_file, N):
    
    # Read the model from the lp file
    model = gurobipy.read(lp_file)
  
    # Set the number of solutions to store in the solution pool
    model.setParam(GRB.Param.PoolSolutions, N)
    
    # Optimize the model
    model.optimize()

    if model.status == GRB.Status.OPTIMAL:
        
        # Get the values of the integer variables
        sol = [v.x for v in model.getVars() if v.vType=='B' and ("U" in v.getAttr("VarName"))]
        # Get the objective value
        obj=model.objVal

    return sol, obj


def makeDataset(filePath,outdir,startIdx,endIdx):
    """
    将(features,adjs,sols,objs)合并制作数据集
    Parameters
    ----------
    filePath : MIP问题的实例文件路径
    outdir :  输出路径
    startIdx,endIdx : 采样起点和终点
    """
    trainFile = os.listdir(filePath)
    if not (os.path.exists(outdir)):
        os.mkdir(outdir)
    adjs=[]
    features=[]
    sols=[]
    objs=[]
    for i in range(startIdx,endIdx):
        
        #获取当前样本的feature和label
        adj,feature=mip_to_bipartite(filePath+trainFile[i])
        sol,obj=solve_mip(filePath+trainFile[i],10)
    
        #邻接矩阵转换为稠密矩阵
        adj=adj.toarray()
        adj=adj.astype(np.uint8)
        
        #归一化
        f_mean=feature.mean()
        f_std=feature.std()
        feature = (feature - f_mean) / f_std
    
        #转换为张量
        adj=torch.from_numpy(adj)
        feature=torch.FloatTensor(feature)
        sol=torch.tensor(sol)
        obj=torch.tensor(obj)
        
        #合并到整个batch
        adjs.append(adj)
        features.append(feature)
        sols.append(sol)
        objs.append(obj)
    
        print('no.{} sample {} '.format(i,trainFile[i]))      
        
        
    #转换为张量
    adjs=torch.stack(adjs)
    features=torch.stack(features)
    sols=torch.stack(sols)
    objs=torch.stack(objs)
    
    #保存
    pickle.dump(adjs,open(outdir+'adjs.pkl','wb'))
    pickle.dump(features,open(outdir+'features.pkl','wb'))
    pickle.dump(sols,open(outdir+'sols.pkl','wb'))
    pickle.dump(objs,open(outdir+'objs.pkl','wb'))

if __name__ == "__main__":
    filePath_train="data/instances/uc/train/"
    outdir1="data/samples/uc/train1/"
    outdir2="data/samples/uc/train2/"
    outdir3="data/samples/uc/train3/"
    outdir4="data/samples/uc/train4/"
    outdir5="data/samples/uc/train5/"
    outdir6="data/samples/uc/train6/"
    outdir7="data/samples/uc/train7/"
    outdir8="data/samples/uc/train8/"
    outdir_valid="data/samples/uc/valid/"
    outdir_test="data/samples/uc/test/"
    '''
    process1=multiprocessing.Process(target=makeDataset,args=(filePath_train,outdir1,0,100))
    process2=multiprocessing.Process(target=makeDataset,args=(filePath_train,outdir2,100,200))
    process3=multiprocessing.Process(target=makeDataset,args=(filePath_train,outdir3,200,300))
    process4=multiprocessing.Process(target=makeDataset,args=(filePath_train,outdir4,300,400))
    process_valid=multiprocessing.Process(target=makeDataset,args=(filePath_train,outdir_valid,500,600))    
    process1.start()
    process2.start()
    process3.start()
    process4.start()
    process_valid.start()
    '''
    process_test=multiprocessing.Process(target=makeDataset,args=(filePath_train,outdir_test,700,800))
    process_test.start()                             
    
    
    
    
    
    
    
    
    
    
    
    