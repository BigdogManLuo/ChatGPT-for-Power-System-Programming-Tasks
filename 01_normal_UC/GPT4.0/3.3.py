from gurobipy import Model, GRB

# Define the parameters
T = 10  # total periods
N = 10  # number of units

# Parameters for the model
C = [19, 18, 19, 20, 16, 19, 24, 18, 11, 23]
S = [290, 276, 210, 296, 299, 243, 260, 255, 222, 227]
D = [101, 120, 147, 101, 192, 125, 112, 179, 171, 191]
P_min = [10, 47, 13, 45, 42, 22, 40, 18, 14, 16]
P_max = [337, 440, 356, 127, 282, 328, 416, 375, 402, 493]
RU = [161, 251, 251, 265, 249, 282, 252, 264, 222, 186]
RD = [161, 251, 251, 265, 249, 282, 252, 264, 222, 186]
U_min = [2,2,2,2,2,2,2,2,2,2]
U_max = [2,2,2,2,2,2,2,2,2,2]
demand = [801, 1655, 483, 1513, 1742, 1034, 1789, 2375, 1289, 1464]
E = [0.5 + 0.1 * i for i in range(N)]
E_max = 50000

# Create the model
m = Model("UnitCommitment")

# Variables
P = m.addVars(N, T, name="P", lb=0)
u = m.addVars(N, T, vtype=GRB.BINARY, name="u")
s = m.addVars(N, T, vtype=GRB.BINARY, name="s")
v = m.addVars(N, T, vtype=GRB.BINARY, name="v")

# Objective Function
m.setObjective(sum(C[i] * P[i,t] + S[i] * s[i,t] + D[i] * v[i,t] for i in range(N) for t in range(T)), GRB.MINIMIZE)

# Constraints
# Power Balance
m.addConstrs((sum(P[i,t] for i in range(N)) == demand[t] for t in range(T)), "PowerBalance")

# Power Output Limits
m.addConstrs((P_min[i] * u[i,t] <= P[i,t] for i in range(N) for t in range(T)), "MinPower")
m.addConstrs((P[i,t] <= P_max[i] * u[i,t] for i in range(N) for t in range(T)), "MaxPower")

# Ramp Constraints
for i in range(N):
    for t in range(1, T):
        m.addConstr(P[i,t] - P[i,t-1] <= RU[i], "RampUp")
        m.addConstr(P[i,t-1] - P[i,t] <= RD[i], "RampDown")

# Minimum On and Off Time
for i in range(N):
    for t in range(2, T):  # start from 2 because we need t-2
        m.addConstr(sum(s[i,k] for k in range(t-U_min[i]+1, t+1)) <= u[i,t], "MinOnTime")
        m.addConstr(sum(v[i,k] for k in range(t-U_max[i]+1, t+1)) <= 1 - u[i,t], "MinOffTime")

# Start-up and Shut-down Definitions
for i in range(N):
    for t in range(1, T):  # start from 1 because we need t-1
        m.addConstr(s[i,t] == u[i,t] - u[i,t-1], "StartupDef")
        m.addConstr(v[i,t] == u[i,t-1] - u[i,t], "ShutdownDef")

# Carbon Emission Constraint
m.addConstrs((sum(E[i] * P[i,t] for i in range(N)) <= E_max for t in range(T)), "CarbonEmission")

# Solve the model
m.optimize()

# Print Results
for i in range(N):
    for t in range(T):
        print(f"Unit {i+1} at time {t+1}: Generated Power = {P[i,t].x}, On/Off = {u[i,t].x}")
