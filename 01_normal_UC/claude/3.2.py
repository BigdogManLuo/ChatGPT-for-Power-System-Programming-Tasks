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
m = Model("unit_commitment")

# Variables
P = {}
u = {}
v = {} 
w = {}
for i in range(N):
    for t in range(T):
        P[i,t] = m.addVar(lb=0, ub=P_max[i], name="P_"+str(i)+"_"+str(t)) 
        u[i,t] = m.addVar(vtype=GRB.BINARY, name="u_"+str(i)+"_"+str(t))
        v[i,t] = m.addVar(vtype=GRB.BINARY, name="v_"+str(i)+"_"+str(t))
        w[i,t] = m.addVar(vtype=GRB.BINARY, name="w_"+str(i)+"_"+str(t))

# Objective
m.setObjective(quicksum(C[i]*P[i,t] + S[i]*v[i,t] + D[i]*w[i,t] for i in range(N) for t in range(T)), GRB.MINIMIZE)

# Constraints  
for t in range(T):
    m.addConstr(quicksum(P[i,t] for i in range(N)) == demand[t]) # Power balance
    
    for i in range(N):
        m.addConstr(P[i,t] >= P_min[i] * u[i,t]) # Min power
        m.addConstr(P[i,t] <= P_max[i] * u[i,t]) # Max power
        
        if t >= U_min[i]:
            m.addConstr(quicksum(v[i,k] for k in range(t-U_min[i]+1, t+1)) <= u[i,t])  # Min up time
        
        if t >= U_max[i]:
            m.addConstr(quicksum(w[i,k] for k in range(t-U_max[i]+1, t+1)) <= 1 - u[i,t]) # Min down time
            
        if t > 0:
            m.addConstr(P[i,t] - P[i,t-1] <= RU[i]) # Ramp up limit
            m.addConstr(P[i,t-1] - P[i,t] <= RD[i]) # Ramp down limit

        if t > 0:  
            m.addConstr(u[i,t] - u[i,t-1] == v[i,t] - w[i,t]) # Startup/shutdown logic
    
    m.addConstr(quicksum(E[i]*P[i,t] for i in range(N)) <= E_max) # Emission limit
            
# Optimize
m.optimize() 

# Print solution
for t in range(T):
    print("Time period", t)
    for i in range(N):
        print("P[%d,%d] = %g" % (i, t, P[i,t].x))
        print("u[%d,%d] = %g" % (i, t, u[i,t].x))
        print("v[%d,%d] = %g" % (i, t, v[i,t].x))
        print("w[%d,%d] = %g" % (i, t, w[i,t].x))
    print()