from gurobipy import Model, GRB

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

model = Model("UnitCommitment")

# Power generation for each unit at each time period
P = model.addVars(N, T, lb=0, name="P")

# Binary status of each unit at each time period
U = model.addVars(N, T, vtype=GRB.BINARY, name="U")

# Binary start-up status of each unit at each time period
S_var = model.addVars(N, T, vtype=GRB.BINARY, name="S_var")

# Binary shut-down status of each unit at each time period
D_var = model.addVars(N, T, vtype=GRB.BINARY, name="D_var")

objective = sum(C[i] * P[i, t] for i in range(N) for t in range(T)) + \
            sum(S[i] * S_var[i, t] for i in range(N) for t in range(T)) + \
            sum(D[i] * D_var[i, t] for i in range(N) for t in range(T))
model.setObjective(objective, GRB.MINIMIZE)

for t in range(T):
    model.addConstr(sum(P[i, t] for i in range(N)) == demand[t], name=f"Power_balance_{t}")

for i in range(N):
    for t in range(T):
        model.addConstr(P_min[i] * U[i, t] <= P[i, t], name=f"Min_Power_{i}_{t}")
        model.addConstr(P[i, t] <= P_max[i] * U[i, t], name=f"Max_Power_{i}_{t}")

for i in range(N):
    for t in range(1, T):
        model.addConstr(sum(U[i, tau] for tau in range(t, min(t + U_min[i], T))) >= U_min[i] * S_var[i, t], name=f"Min_on_time_{i}_{t}")
        model.addConstr(sum((1 - U[i, tau]) for tau in range(t, min(t + U_max[i], T))) >= U_max[i] * D_var[i, t], name=f"Min_off_time_{i}_{t}")

for i in range(N):
    for t in range(1, T):
        model.addConstr(P[i, t] - P[i, t - 1] <= RU[i], name=f"Ramp_up_{i}_{t}")
        model.addConstr(P[i, t - 1] - P[i, t] <= RD[i], name=f"Ramp_down_{i}_{t}")

for i in range(N):
    for t in range(1, T):
        model.addConstr(U[i, t] - U[i, t - 1] == S_var[i, t] - D_var[i, t], name=f"Startup_Shutdown_{i}_{t}")

for t in range(T):
    model.addConstr(sum(E[i] * P[i, t] for i in range(N)) <= E_max, name=f"Carbon_Emission_{t}")

model.optimize()

if model.status == GRB.OPTIMAL:
    for i in range(N):
        for t in range(T):
            print(f"Unit {i} at time {t}: Production = {P[i, t].x}, Status = {U[i, t].x}, Start = {S_var[i, t].x}, Shutdown = {D_var[i, t].x}")
