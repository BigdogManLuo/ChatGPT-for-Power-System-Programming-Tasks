from gurobipy import *

# Define parameters
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
m.setObjective(sum(C[n] * P[n,t] + S[n] * V[n,t] + D[n] * W[n,t] for n in range(N) for t in range(T)), GRB.MINIMIZE)

# Add power balance constraint
m.addConstrs((sum(P[n,t] for n in range(N)) == demand[t] for t in range(T)), "power_balance")

# Add power output limit constraint
for n in range(N):
    for t in range(T):
        m.addConstr(P[n,t] >= P_min[n] * U[n,t], "lower_bound_{}_{}".format(n, t))
        m.addConstr(P[n,t] <= P_max[n] * U[n,t], "upper_bound_{}_{}".format(n, t))

# Add minimum start-stop time constraint
for n in range(N):
    for t in range(T - max(U_min[n], U_max[n]) + 1):
        m.addConstr(sum(U[n,k] for k in range(t, t + U_min[n])) >= U_min[n] * V[n,t], "min_start_time_{}_{}".format(n, t))
        m.addConstr(sum((1 - U[n,k]) for k in range(t, t + U_max[n])) >= U_max[n] * W[n,t], "min_down_time_{}_{}".format(n, t))

# Add ramp power constraint
for n in range(N):
    for t in range(1, T):
        m.addConstr(P[n,t] - P[n,t-1] <= RU[n] * (1 - W[n,t]), "ramp_up_{}_{}".format(n, t))
        m.addConstr(P[n,t-1] - P[n,t] <= RD[n] * (1 - V[n,t]), "ramp_down_{}_{}".format(n, t))

# Add start-up and shut-down variables
for n in range(N):
    for t in range(1, T):
        m.addConstr(V[n,t] >= U[n,t] - U[n,t-1], "start_var_{}_{}".format(n, t))
        m.addConstr(W[n,t] >= U[n,t-1] - U[n,t], "shut_var_{}_{}".format(n, t))

# Add carbon emission constraint
m.addConstr(sum(E[n] * P[n,t] for n in range(N) for t in range(T)) <= E_max, "carbon_emission")

# Optimize model
m.optimize()

# Print results
if m.status == GRB.Status.OPTIMAL:
    for n in range(N):
        for t in range(T):
            print('Generator', n, 'at time', t, ':', 'Power=', P[n,t].x, 'On/Off=', U[n,t].x)
else:
    print('No solution')
