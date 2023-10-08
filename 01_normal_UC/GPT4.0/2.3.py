from gurobipy import Model, GRB

# Parameter initialization
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
m = Model('uc')

# Create variables
P = m.addVars(N, T, lb=0, name="P")  # Power output
x = m.addVars(N, T, vtype=GRB.BINARY, name="x")  # Binary variable for unit on/off
u = m.addVars(N, T, vtype=GRB.BINARY, name="u")  # Binary variable for start up
v = m.addVars(N, T, vtype=GRB.BINARY, name="v")  # Binary variable for shut down

# Set objective
m.setObjective(sum(C[i] * P[i,t] + S[i] * u[i,t] + D[i] * v[i,t] for i in range(N) for t in range(T)), GRB.MINIMIZE)

# Power balance constraints
m.addConstrs((sum(P[i,t] for i in range(N)) == demand[t] for t in range(T)), "power_balance")

# Power output limit constraints
m.addConstrs((P[i,t] >= P_min[i] * x[i,t] for i in range(N) for t in range(T)), "min_power_output")
m.addConstrs((P[i,t] <= P_max[i] * x[i,t] for i in range(N) for t in range(T)), "max_power_output")

# Minimum up/down time constraints
for i in range(N):
    for t in range(T):
        if t < T - U_min[i] + 1:
            m.addConstr(sum(x[i, tt] for tt in range(t, t + U_min[i])) >= U_min[i] * u[i, t])
        if t < T - U_max[i] + 1:
            m.addConstr(sum((1 - x[i, tt]) for tt in range(t, t + U_max[i])) >= U_max[i] * v[i, t])

# Ramp up/down constraints
m.addConstrs((P[i,t] - P[i,t-1] <= RU[i] for i in range(N) for t in range(1, T)), "ramp_up")
m.addConstrs((P[i,t-1] - P[i,t] <= RD[i] for i in range(N) for t in range(1, T)), "ramp_down")

# Carbon emission constraint
m.addConstr(sum(E[i] * P[i,t] for i in range(N) for t in range(T)) <= E_max, "emission")

# Unit status transition constraints
m.addConstrs((x[i,t] - x[i,t-1] == u[i,t] - v[i,t] for i in range(N) for t in range(1, T)), "unit_transition")

# Solve
m.optimize()
