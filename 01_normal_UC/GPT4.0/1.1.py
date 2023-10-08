from gurobipy import *

# Define the parameters
T = 10  # total periods
N = 10  # number of units
# Define parameters for the model
C = [19, 18, 19, 20, 16, 19, 24, 18, 11, 23]  # Cost coefficients
S = [290, 276, 210, 296, 299, 243, 260, 255, 222, 227]  # Start-up costs
DU = [101, 120, 147, 101, 192, 125, 112, 179, 171, 191]  # Shutdown costs
P_min = [10, 47, 13, 45, 42, 22, 40, 18, 14, 16]  # Min power output
P_max = [337, 440, 356, 127, 282, 328, 416, 375, 402, 493]  # Max power output
RU = [161, 251, 251, 265, 249, 282, 252, 264, 222, 186]  # Ramp-up rate
RD = [161, 251, 251, 265, 249, 282, 252, 264, 222, 186]  # Ramp-down rate
U_min = [2,2,2,2,2,2,2,2,2,2]  # Minimum up time
U_max = [2,2,2,2,2,2,2,2,2,2]  # Minimum down time
demand = [801, 1655, 483, 1513, 1742, 1034, 1789, 2375, 1289, 1464]  # Load demand
E = [0.5 + 0.1 * i for i in range(N)]  # Carbon emissions rate
E_max = 50000  # Maximum total carbon emissions

# Initialize a new model
model = Model("Unit Commitment")

# Decision variables
G = model.addVars(N, T, lb=0.0, name="G")  # generation
X = model.addVars(N, T, vtype=GRB.BINARY, name="X")  # status
U = model.addVars(N, T, vtype=GRB.BINARY, name="U")  # startup
D = model.addVars(N, T, vtype=GRB.BINARY, name="D")  # shutdown

# Objective function
model.setObjective(quicksum(C[i]*G[i,t] for i in range(N) for t in range(T)) + 
                   quicksum(S[i]*U[i,t] for i in range(N) for t in range(T)) + 
                   quicksum(DU[i]*D[i,t] for i in range(N) for t in range(T)), GRB.MINIMIZE)

# Power balance constraint
model.addConstrs((quicksum(G[i,t] for i in range(N)) == demand[t] for t in range(T)), name="Power_balance")

# Generation limit constraints
model.addConstrs((G[i,t] >= P_min[i]*X[i,t] for i in range(N) for t in range(T)), name="Min_gen")
model.addConstrs((G[i,t] <= P_max[i]*X[i,t] for i in range(N) for t in range(T)), name="Max_gen")

# Ramp rate constraints
model.addConstrs((G[i,t] - G[i,t-1] <= RU[i]*X[i,t] for i in range(N) for t in range(1,T)), name="Ramp_up")
model.addConstrs((G[i,t-1] - G[i,t] <= RD[i]*X[i,t] for i in range(N) for t in range(1,T)), name="Ramp_down")

# Minimum up/down time constraints
for i in range(N):
    for t in range(T):
        model.addConstr(quicksum(X[i,k] for k in range(t,min(t+U_min[i],T))) >= U_min[i]*U[i,t], name=f"Min_up_{i}_{t}")
        model.addConstr(quicksum(1-X[i,k] for k in range(t,min(t+U_max[i],T))) >= U_max[i]*D[i,t], name=f"Min_down_{i}_{t}")

# Startup and shutdown conditions
model.addConstrs((U[i,t] >= X[i,t] - X[i,t-1] for i in range(N) for t in range(1,T)), name="Startup")
model.addConstrs((D[i,t] >= X[i,t-1] - X[i,t] for i in range(N) for t in range(1,T)), name="Shutdown")

# Carbon emissions constraint
model.addConstr((quicksum(E[i]*G[i,t] for i in range(N) for t in range(T)) <= E_max), name="Emissions")

# Solve the model
model.optimize()

# Output the results
if model.status == GRB.OPTIMAL:
    print('Optimal solution found')
    for v in model.getVars():
        print(f'{v.varName} = {v.x}')
else:
    print('No optimal solution found')
