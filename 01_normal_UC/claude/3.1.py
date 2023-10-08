import gurobipy as gp
from gurobipy import *

# Define the parameters
T = 10 # total periods
N = 10 # number of units
# Define parameters for the model
C = [19, 18, 19, 20, 16, 19, 24, 18, 11, 23] # Cost coefficients
S = [290, 276, 210, 296, 299, 243, 260, 255, 222, 227] # Start-up costs
D = [101, 120, 147, 101, 192, 125, 112, 179, 171, 191] # Shutdown costs
P_min = [10, 47, 13, 45, 42, 22, 40, 18, 14, 16] # Min power output
P_max = [337, 440, 356, 127, 282, 328, 416, 375, 402, 493] # Max power output
RU = [161, 251, 251, 265, 249, 282, 252, 264, 222, 186] # Ramp-up rate
RD = [161, 251, 251, 265, 249, 282, 252, 264, 222, 186] # Ramp-down rate
U_min = [2,2,2,2,2,2,2,2,2,2] # Minimum up time
U_max = [2,2,2,2,2,2,2,2,2,2] # Minimum down time
demand = [801, 1655, 483, 1513, 1742, 1034, 1789, 2375, 1289, 1464] # Load demand
E = [0.5 + 0.1 * i for i in range(N)] # Carbon emissions rate
E_max = 50000 # Maximum total carbon emissions

# Model
m = gp.Model("unit_commitment")

# Create variables
P = {}
y = {} 
for i in range(N):
    for t in range(T):
        P[i,t] = m.addVar(lb=0, ub=GRB.INFINITY, name="P_%s_%s" % (i, t))
        y[i,t] = m.addVar(vtype=GRB.BINARY, name="y_%s_%s" % (i, t))

z = {}
w = {}
for i in range(N):
    for t in range(T):
        if t > 0:
            z[i,t] = m.addVar(vtype=GRB.BINARY, name="z_%s_%s" % (i, t))
            w[i,t] = m.addVar(vtype=GRB.BINARY, name="w_%s_%s" % (i, t))

# Objective
obj = quicksum(C[i]*P[i,t] for i in range(N) for t in range(T))
for t in range(1,T):
    obj += quicksum(S[i]*z[i,t] + D[i]*w[i,t] for i in range(N)) 
m.setObjective(obj, GRB.MINIMIZE)

# Constraints  
# Power balance
for t in range(T):
    m.addConstr(quicksum(P[i,t] for i in range(N)) == demand[t])
    
# Generation limits  
for i in range(N):
    for t in range(T):
        m.addConstr(P_min[i] * y[i,t] <= P[i,t]) 
        m.addConstr(P[i,t] <= P_max[i] * y[i,t])
        
# Ramp limits
for i in range(N):
    for t in range(1,T):
        m.addConstr(P[i,t] - P[i,t-1] <= RU[i])
        m.addConstr(P[i,t-1] - P[i,t] <= RD[i])
        
# Minimum up time        
for i in range(N):
    for t in range(U_min[i]-1, T):
        m.addConstr(quicksum(y[i,k] for k in range(t-U_min[i]+1, t+1)) >= U_min[i]*z[i,t])
        
# Minimum down time
for i in range(N):
    for t in range(U_max[i]-1, T):
        m.addConstr(quicksum(1 - y[i,k] for k in range(t-U_max[i]+1, t+1)) >= U_max[i]*w[i,t])
        
# Startup/shutdown logic        
for i in range(N):
    for t in range(1,T):
        m.addConstr(y[i,t] - y[i,t-1] == z[i,t] - w[i,t])
        
# Emission limit
for t in range(T):
    m.addConstr(quicksum(E[i]*P[i,t] for i in range(N)) <= E_max)
    
# Optimize
m.optimize()

# Print solution
for i in range(N):
    for t in range(T):
        if y[i,t].x > 0.5:
            print("Generator %s is ON in period %s" % (i, t))
        print("P[%s,%s] = %s" % (i, t, P[i,t].x))