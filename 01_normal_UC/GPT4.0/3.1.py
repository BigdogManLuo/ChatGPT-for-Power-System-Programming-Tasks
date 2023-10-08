from gurobipy import Model, GRB, quicksum

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

# Create a model
m = Model("UnitCommitment")

# Define the variables
P = [[m.addVar(vtype=GRB.CONTINUOUS) for t in range(T)] for i in range(N)]  # Power output
u = [[m.addVar(vtype=GRB.BINARY) for t in range(T)] for i in range(N)]  # Unit status
s = [[m.addVar(vtype=GRB.BINARY) for t in range(T)] for i in range(N)]  # Startup variable
d = [[m.addVar(vtype=GRB.BINARY) for t in range(T)] for i in range(N)]  # Shutdown variable

# Define the objective
m.setObjective(quicksum(C[i] * P[i][t] + S[i] * s[i][t] + D[i] * d[i][t] for i in range(N) for t in range(T)), GRB.MINIMIZE)

# Add power balance constraint
for t in range(T):
    m.addConstr(quicksum(P[i][t] for i in range(N)) == demand[t])

# Add power output limit, start-stop variable update, ramp-up and ramp-down constraints
for i in range(N):
    for t in range(T):
        m.addConstr(P_min[i] * u[i][t] <= P[i][t])
        m.addConstr(P[i][t] <= P_max[i] * u[i][t])
        if t > 0:
            m.addConstr(u[i][t] - u[i][t-1] == s[i][t] - d[i][t])
            m.addConstr(P[i][t] - P[i][t-1] <= RU[i])
            m.addConstr(P[i][t-1] - P[i][t] <= RD[i])

# Add minimum on and off time constraints
for i in range(N):
    for t in range(T):
        if t >= U_min[i]:
            m.addConstr(quicksum(s[i][t-k] for k in range(U_min[i])) <= u[i][t])
        if t >= U_max[i]:
            m.addConstr(quicksum(d[i][t-k] for k in range(U_max[i])) <= 1 - u[i][t])

# Add carbon emission constraint
m.addConstr(quicksum(E[i] * P[i][t] for i in range(N) for t in range(T)) <= E_max)

# Optimize
m.optimize()

# Print the optimal schedule
for i in range(N):
    print("Unit", i, ":")
    for t in range(T):
        print("Period", t, ": Power =", P[i][t].X, " Status =", u[i][t].X)
