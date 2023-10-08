import gurobipy as gp
from gurobipy import GRB, quicksum

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

# Sets
I = range(N) # Generator index set  
T = range(T) # Time period index set

# Variables
u = m.addVars(I,T, vtype=GRB.BINARY, name="u") # Commitment status
v = m.addVars(I,T, vtype=GRB.BINARY, name="v") # Startup vars
w = m.addVars(I,T, vtype=GRB.BINARY, name="w") # Shutdown vars 
P = m.addVars(I,T, name="P") # Power generation

# Objective
m.setObjective(
    quicksum(C[i]*P[i,t] for i in I for t in T) +  
    quicksum(S[i]*v[i,t] + D[i]*w[i,t] for i in I for t in T),
    GRB.MINIMIZE)


# Constraints
# Power balance
for t in T:
    m.addConstr(P.sum('*',t) == demand[t], "load_balance") 
    
# Min/max limits
m.addConstrs((P_min[i]*u[i,t] <= P[i,t] for i in I for t in T), "min_power")
m.addConstrs((P[i,t] <= P_max[i]*u[i,t] for i in I for t in T), "max_power")

# Ramp limits
m.addConstrs((P[i,t] - P[i,t-1] <= RU[i]*u[i,t] for i in I for t in T if t>0), "ramp_up")
m.addConstrs((P[i,t-1] - P[i,t] <= RD[i]*u[i,t] for i in I for t in T if t>0), "ramp_down")

# Minimum uptime
m.addConstrs((v[i,t] <= u[i,t] for i in I for t in T), "min_uptime1")  
m.addConstrs((quicksum(v[i,t] for t in range(max(0,t-U_min[i]+1),t+1)) <= 1 for i in I for t in T), 
             "min_uptime2")
             
# Minimum downtime             
m.addConstrs((w[i,t] <= 1 - u[i,t] for i in I for t in T), "min_downtime1")
m.addConstrs((quicksum(w[i,t] for t in range(max(0,t-U_max[i]+1),t+1)) <= 1 for i in I for t in T),
             "min_downtime2")
             
# Carbon emission limit
m.addConstr(quicksum(E[i]*P[i,t] for i in I for t in T) <= E_max, "emission_limit")             

# Solve            
m.optimize() 

# Print solution
for v in m.getVars():
    print('%s %g' % (v.varName, v.x))

print('Obj: %g' % m.objVal)