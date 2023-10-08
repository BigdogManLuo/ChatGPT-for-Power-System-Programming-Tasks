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

# Create model
model = Model("unit_commitment")

# Sets
I = range(N) 
T = range(T)

# Variables
p = {}
u = {} 
v = {}
w = {}
for i in I:
    for t in T:
        p[i,t] = model.addVar(lb=0, ub=GRB.INFINITY, name=f'p_{i}_{t}')
        u[i,t] = model.addVar(vtype=GRB.BINARY, name=f'u_{i}_{t}') 
        v[i,t] = model.addVar(vtype=GRB.BINARY, name=f'v_{i}_{t}')
        w[i,t] = model.addVar(vtype=GRB.BINARY, name=f'w_{i}_{t}')

# Objective 
model.setObjective(quicksum(C[i]*p[i,t] + S[i]*v[i,t] + D[i]*w[i,t] for i in I for t in T)) 

# Constraints
for t in T:
    # Power balance 
    model.addConstr(quicksum(p[i,t] for i in I) == demand[t]) 
    
    # Min/max limits
    for i in I:
        model.addConstr(P_min[i]*u[i,t] <= p[i,t]) 
        model.addConstr(p[i,t] <= P_max[i]*u[i,t])

# Ramp limits
for i in I:
    for t in T[1:]:
        model.addConstr(p[i,t] - p[i,t-1] <= RU[i])
        model.addConstr(p[i,t-1] - p[i,t] <= RD[i])
        
# Min up/down time        
for i in I:
    for t in T[U_min[i]-1:]:
        model.addConstr(u[i,t] - u[i,t-1] <= v[i,t])
        model.addConstr(quicksum(u[i,tau] for tau in range(t-U_min[i]+1, t+1)) >= U_min[i]*v[i,t])

    for t in T[U_max[i]-1:]:
        model.addConstr(u[i,t-1] - u[i,t] <= w[i,t])
        model.addConstr(quicksum(1 - u[i,tau] for tau in range(t-U_max[i]+1, t+1)) >= U_max[i]*w[i,t])
        
# Emission limit        
for t in T:
    model.addConstr(quicksum(E[i]*p[i,t] for i in I) <= E_max)
    
# Solve
model.optimize()