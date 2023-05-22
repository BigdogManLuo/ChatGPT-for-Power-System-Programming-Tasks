from gurobipy import *
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os

# Define the parameters
T = 24  # total periods
N = 5  # number of units

#Sample Setting
sampleNumber=1000

for k in range(sampleNumber):

    # Define parameters for the model.
    C = [19, 18, 19, 26, 20]  # Cost coefficients
    S = [290, 276, 210, 296, 299]  # Start-up costs
    D = [101, 120, 147, 101, 192] # Shutdown costs
    P_min = [10, 47, 13, 45, 42]  # Min power output
    P_max = [337, 440, 356, 127, 282]  # Max power output
    RU = [161, 251, 251, 265, 249] # Ramp-up rate
    RD = [161, 251, 251, 265, 249]  # Ramp-down rate
    U_min = [2,2,2,2,2] # Minimum up time
    U_max = [2,2,2,2,2]   # Minimum down time
    demand = [876, 892, 645, 927, 683, 789, 861, 969, 753, 785,927, 640, 658, 930, 811, 686, 743, 943, 938, 983,925, 635, 615, 659]  # Load demand
    M = 1  # Big M
    
    #添加随机数
    C = [num + np.random.randint(0,2) for num in C]
    S = [num + np.random.randint(0,2) for num in S]
    D = [num + np.random.randint(0,2) for num in D]
    P_min = [num + np.random.randint(0,2) for num in P_min]
    P_max = [num + np.random.randint(0,5) for num in P_max]
    RU = [num + np.random.randint(0,2) for num in RU]
    RD = [num + np.random.randint(0,2) for num in RD]
    demand=[num + np.random.randint(0,5) for num in demand]
    
    # Create a new model
    m = Model("UC")
    
    # Create variables
    P = [[m.addVar(lb=0, ub=P_max[i], name="P_{}_{}".format(i, t)) for t in range(T)] for i in range(N)]  # Power output
    U = [[m.addVar(vtype=GRB.BINARY, name="U_{}_{}".format(i, t)) for t in range(T)] for i in range(N)]  # Unit status
    V = [[m.addVar(vtype=GRB.BINARY, name="V_{}_{}".format(i, t)) for t in range(T)] for i in range(N)]  # Start-up status
    W = [[m.addVar(vtype=GRB.BINARY, name="W_{}_{}".format(i, t)) for t in range(T)] for i in range(N)]  # Shutdown status
    
    # Set objective
    m.setObjective(sum(C[i]*P[i][t] + S[i]*V[i][t] + D[i]*W[i][t] for i in range(N) for t in range(T)), GRB.MINIMIZE)
    
    # Add constraints
    for t in range(T):
        # Power balance constraint
        m.addConstr(sum(P[i][t] for i in range(N)) == demand[t], "PowerBalance_{}".format(t))
        
        for i in range(N):
            # Unit status constraint
            m.addConstr(P_min[i]*U[i][t] <= P[i][t], "MinPower_{}_{}".format(i, t))
            m.addConstr(P[i][t] <= P_max[i]*U[i][t], "MaxPower_{}_{}".format(i, t))
    
            # Startup and shutdown constraints
            if t > 0:
                m.addConstr(V[i][t] >= U[i][t] - U[i][t-1], "Startup1_{}_{}".format(i, t))
                m.addConstr(V[i][t] <= U[i][t] - U[i][t-1] + M*(1 - U[i][t-1]), "Startup2_{}_{}".format(i, t))
    
                m.addConstr(W[i][t] >= U[i][t-1] - U[i][t], "Shutdown1_{}_{}".format(i, t))
                m.addConstr(W[i][t] <= U[i][t-1] - U[i][t] + M*(1 - U[i][t]), "Shutdown2_{}_{}".format(i, t))
    
                # Ramp constraints
                m.addConstr(P[i][t] - P[i][t-1] <= RU[i], "RampUp_{}_{}".format(i, t))
                m.addConstr(P[i][t-1] - P[i][t] <= RD[i], "RampDown_{}_{}".format(i, t))
    
    # Add minimum up/down time constraints
    '''
    for i in range(N):
        for t in range(U_min[i], T):
            m.addConstr(sum(V[i][t-j] for j in range(U_min[i])) <= 1, "MinUp_{}_{}".format(i, t))
        for t in range(U_max[i], T):
            m.addConstr(sum(W[i][t-j] for j in range(U_max[i])) <= 1, "MinDown_{}_{}".format(i, t))
    '''
    
    
    # 写入LP文件
    m.write(outdir+"instance"+str(k)+".lp")
    
    #进度显示
    print(str(k)+"/"+str(sampleNumber))

