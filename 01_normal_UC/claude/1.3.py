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
U_min = [2,2,2,2,2,2,2,2,2,2]
D_min = [2,2,2,2,2,2,2,2,2,2] # Minimum down time
demand = [801, 1655, 483, 1513, 1742, 1034, 1789, 2375, 1289, 1464]
E = [0.5 + 0.1 * i for i in range(N)]
E_max = 50000

# Create model
m = gp.Model("unit_commitment")

# Decision variables
P = m.addVars(N,T, name="P") # Power output
y = m.addVars(N,T, vtype=GRB.BINARY, name="y") # Unit on/off
v = m.addVars(N,T, vtype=GRB.BINARY, name="v") # Startup 
w = m.addVars(N,T, vtype=GRB.BINARY, name="w") # Shutdown
E_tot = m.addVars(T, name="E_tot") # Total emissions

# Objective function
m.setObjective(gp.quicksum(C[i]*P[i,t] + S[i]*v[i,t] + D[i]*w[i,t] for i in range(N) for t in range(T)) , GRB.MINIMIZE)

# Constraints
for t in range(T):
    m.addConstr(gp.quicksum(P[i,t] for i in range(N)) == demand[t]) # Power balance
    m.addConstr(E_tot[t] == gp.quicksum(E[i]*P[i,t] for i in range(N))) # Emissions     

# Startup constraints
for i in range(N):
   for t in range(T):
      if t > 0:
         m.addConstr(v[i,t] >= y[i,t] - y[i,t-1]) # Link startup to on/off status

# Shutdown constraints       
for i in range(N):
   for t in range(T):
      if t > 0:
         m.addConstr(w[i,t] >= y[i,t-1] - y[i,t]) # Link shutdown to on/off status

for i in range(N):
    for t in range(T):
        m.addConstr(P_min[i] * y[i,t] <= P[i,t]) # Min power
        m.addConstr(P[i,t] <= P_max[i] * y[i,t]) # Max power
        
        if t > 0:
            m.addConstr(P[i,t] - P[i,t-1] <= RU[i]) # Ramp up limit
            m.addConstr(P[i,t-1] - P[i,t] <= RD[i]) # Ramp down limit
            
        if t >= U_min[i]:
            m.addConstr(gp.quicksum(v[i,k] for k in range(t-U_min[i], t+1)) <= y[i,t]) # Min up time
            
        if t >= D_min[i]:
            m.addConstr(gp.quicksum(w[i,k] for k in range(t-D_min[i], t+1)) >= 1 - y[i,t]) # Min down time
            
m.optimize()