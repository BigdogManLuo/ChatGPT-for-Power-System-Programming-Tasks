import gurobipy as gp 
from gurobipy import GRB

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

# Initialize decision variables
p = {}
u = {} 
v = {}
w = {}
for i in range(N):
    for t in range(T):
        p[i,t] = m.addVar(name="p_%d_%d"%(i,t)) 
        u[i,t] = m.addVar(vtype=GRB.BINARY, name="u_%d_%d"%(i,t))
        v[i,t] = m.addVar(vtype=GRB.BINARY, name="v_%d_%d"%(i,t))
        w[i,t] = m.addVar(vtype=GRB.BINARY, name="w_%d_%d"%(i,t))

# Objective function
m.setObjective(gp.quicksum(C[i]*p[i,t] + S[i]*v[i,t] + D[i]*w[i,t] for i in range(N) for t in range(T)), GRB.MINIMIZE)

# Constraints
for t in range(T):
    # Power balance 
    m.addConstr(gp.quicksum(p[i,t] for i in range(N)) == demand[t]) 
    
    # Min/max limits
    for i in range(N):
        m.addConstr(P_min[i] * u[i,t] <= p[i,t])
        m.addConstr(p[i,t] <= P_max[i] * u[i,t])
        
    # Ramp up/down limits
    for i in range(N):
        if t > 0:
            m.addConstr(p[i,t] - p[i,t-1] <= RU[i]) 
            m.addConstr(p[i,t-1] - p[i,t] <= RD[i])
            
    # Min up/down time
    for i in range(N):
        if t > 0:
            m.addConstr(u[i,t] >= u[i,t-1] - w[i,t-1])
            m.addConstr(u[i,t] <= u[i,t-1] + v[i,t-1])
        
    for i in range(N):
        for t in range(U_min[i]-1, T):
            m.addConstr(gp.quicksum(w[i,j] for j in range(t-U_min[i]+1, t+1)) <= 1-u[i,t])

        for i in range(N):
            for t in range(U_max[i]-1, T):
                m.addConstr(gp.quicksum(v[i,j] for j in range(t-U_max[i]+1, t+1)) <= u[i,t])

    # Startup/shutdown 
    for i in range(N):
        m.addConstr(v[i,t] + w[i,t] <= 1)
        
    # Emission limit
    m.addConstr(gp.quicksum(E[i] * p[i,t] for i in range(N)) <= E_max)
    
# Optimize    
m.optimize()

# Print solution
for i in range(N):
    for t in range(T):
        if p[i,t].x > 0:
            print("Generator %d produces %g MW in period %d"%(i,p[i,t].x,t))

print("Total cost = ", m.objVal)

