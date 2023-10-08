from gurobipy import *

# Define the parameters
T = 10  # total periods
N = 10  # number of units
# Define parameters for the model
C = [19, 18, 19, 20, 16, 19, 24, 18, 11, 23]  # Cost coefficients
S = [290, 276, 210, 296, 299, 243, 260, 255, 222, 227]  # Start-up costs
D = [101, 120, 147, 101, 192, 125, 112, 179, 171, 191]  # Shutdown costs
P_min = [10, 47, 13, 45, 42, 22, 40, 18, 14, 16]  # Min power output
P_max = [337, 440, 356, 127, 282, 328, 416, 375, 402, 493]  # Max power output
RU = [161, 251, 251, 265, 249, 282, 252, 264, 222, 186]  # Ramp-up rate
RD = [161, 251, 251, 265, 249, 282, 252, 264, 222, 186]  # Ramp-down rate
U_min = [2,2,2,2,2,2,2,2,2,2]  # Minimum up time
U_max = [2,2,2,2,2,2,2,2,2,2]  # Minimum down time
demand = [801, 1655, 483, 1513, 1742, 1034, 1789, 2375, 1289, 1464]  # Load demand
E = [0.5 + 0.1 * i for i in range(N)]  # Carbon emissions rate
E_max = 50000  # Maximum total carbon emissions

# Create a new model
m = Model("uc")

# Create variables
P = m.addVars(N, T, lb=0, vtype=GRB.CONTINUOUS, name="P")
U = m.addVars(N, T, vtype=GRB.BINARY, name="U")
V = m.addVars(N, T, vtype=GRB.BINARY, name="V")
W = m.addVars(N, T, vtype=GRB.BINARY, name="W")

# Set objective
m.setObjective(quicksum((C[i]*P[i,t] + S[i]*V[i,t] + D[i]*W[i,t]) for i in range(N) for t in range(T)), GRB.MINIMIZE)

# Add power balance constraint
m.addConstrs((quicksum(P[i,t] for i in range(N)) == demand[t] for t in range(T)), "power_balance")

# Add power output limit constraint
for i in range(N):
    for t in range(T):
        m.addConstr(P[i,t] <= P_max[i]*U[i,t], "max_power")
        m.addConstr(P[i,t] >= P_min[i]*U[i,t], "min_power")

# Add minimum start-stop time constraint
for i in range(N):
    for t in range(3, T):
        m.addConstr(quicksum(V[i,k] for k in range(t-U_min[i]+1, t+1)) <= U[i,t], "min_up_time")
        m.addConstr(quicksum(W[i,k] for k in range(t-U_max[i]+1, t+1)) <= 1-U[i,t], "min_down_time")

# Add ramp power constraint
for i in range(N):
    for t in range(1, T):
        m.addConstr(P[i,t] - P[i,t-1] <= RU[i]*U[i,t-1], "ramp_up")
        m.addConstr(P[i,t] - P[i,t-1] >= -RD[i]*U[i,t], "ramp_down")

# Add carbon emission constraint
m.addConstr(quicksum(E[i]*P[i,t] for i in range(N) for t in range(T)) <= E_max, "emission")

# Add binary variables constraint
for i in range(N):
    for t in range(1, T):
        m.addConstr(U[i,t-1] - U[i,t] == V[i,t] - W[i,t], "on_off_status")
        m.addConstr(V[i,t] + W[i,t] <= 1, "start_up_shutdown")

# Solve model
m.optimize()

# Print the optimal solution
for v in m.getVars():
    print(f"{v.varName}: {v.x}")

print(f"Optimal cost: {m.objVal}")
