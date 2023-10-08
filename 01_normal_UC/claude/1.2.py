import gurobipy as gp
from gurobipy import GRB

# Model parameters
T = 10 
N = 10
C = [19, 18, 19, 20, 16, 19, 24, 18, 11, 23] 
S = [290, 276, 210, 296, 299, 243, 260, 255, 222, 227]
D = [101, 120, 147, 101, 192, 125, 112, 179, 171, 191]
P_min = [10, 47, 13, 45, 42, 22, 40, 18, 14, 16]
P_max = [337, 440, 356, 127, 282, 328, 416, 375, 402, 493]
RU = [161, 251, 251, 265, 249, 282, 252, 264, 222, 186]
RD = [161, 251, 251, 265, 249, 282, 252, 264, 222, 186] 
UT = [2,2,2,2,2,2,2,2,2,2]
DT = [2,2,2,2,2,2,2,2,2,2]
demand = [801, 1655, 483, 1513, 1742, 1034, 1789, 2375, 1289, 1464]
E = [0.5 + 0.1 * i for i in range(N)]
E_max = 50000

# Create model
m = gp.Model('UnitCommitment')

# Create variables
P = {}
u = {}
for i in range(N):
    for t in range(T):
        P[i,t] = m.addVar(lb=0, ub=P_max[i], vtype=GRB.CONTINUOUS, name=f'P_{i}_{t}') 
        u[i,t] = m.addVar(vtype=GRB.BINARY, name=f'u_{i}_{t}')

# Objective function        
obj = gp.quicksum(C[i]*P[i,t] for i in range(N) for t in range(T))
obj += gp.quicksum(S[i]*u[i,t] for i in range(N) for t in range(T)) 
obj += gp.quicksum(D[i]*(1-u[i,t]) for i in range(N) for t in range(T))
m.setObjective(obj, GRB.MINIMIZE)

# Constraints
for t in range(T):
    m.addConstr(gp.quicksum(P[i,t] for i in range(N)) >= demand[t]) 
    
for i in range(N):
    for t in range(T):
        if t < UT[i]: 
            m.addConstr(gp.quicksum(u[i,k] for k in range(t+1)) >= u[i,t])
        if t < DT[i]:
            m.addConstr(gp.quicksum(1 - u[i,k] for k in range(t+1)) >= 1 - u[i,t])
            
for i in range(N):    
    for t in range(1,T):
        m.addConstr(P[i,t] - P[i,t-1] <= RU[i])
        m.addConstr(P[i,t-1] - P[i,t] <= RD[i])
        
for i in range(N):    
    for t in range(T):
        m.addConstr(P_min[i] * u[i,t] <= P[i,t])
        m.addConstr(P[i,t] <= P_max[i] * u[i,t])
        
m.addConstr(gp.quicksum(E[i]*P[i,t] for i in range(N) for t in range(T)) <= E_max)

# Optimize
m.optimize() 

# Print solution
for v in m.getVars():
    print(v.varName, v.x)

print("Total cost =", obj.getValue())