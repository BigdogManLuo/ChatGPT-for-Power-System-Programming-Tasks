import gurobipy as gp

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

# Create the model
model = gp.Model("unit_commitment")

# Create the decision variables
P = model.addVars(T, N, vtype=gp.GRB.CONTINUOUS)  # Power output
s = model.addVars(T, N, vtype=gp.GRB.BINARY)  # On/off status

# Objective function
obj = gp.LinExpr()
for t in range(T):
    for i in range(N):
        obj += C[i] * P[t, i] + S[i] * s[t, i] + D[i] * (1 - s[t, i])
obj += gp.LinExpr(
    [E[i] * P[t, i] for i in range(N)],
    name="Carbon emission",
)
model.setObjective(obj, gp.GRB.MINIMIZE)

# Constraints
for t in range(T):
    model.addConstr(gp.sum(P[t, i] for i in range(N)) == demand[t])
    for i in range(N):
        model.addConstr(P[t, i] <= P_max[i])
        model.addConstr(P[t, i] >= P_min[i])
        model.addConstr(P[t, i] - P[t - 1, i] <= RU[i])
        model.addConstr(P[t, i] - P[t - 1, i] >= -RD[i])
        model.addConstr(s[t, i] - s[t - 1, i] <= 1 - U_min[i])
        model.addConstr(s[t, i] - s[t - 1, i] >= -1 + U_max[i])
        model.addConstr(s[t, i] <= 1)
        model.addConstr(s[t, i] >= 0)

# Carbon emission constraint
model.addConstr(gp.sum(E[i] * P[t, i] for i in range(N)) <= E_max)

# Optimize the model
model.optimize()

# Print the solution
if model.status == gp.GRB.OPTIMAL:
    print("Optimal objective value:", model.objVal)
    for t in range(T):
        for i in range(N):
            print("Power output at time t=%d, unit %d: %f" % (t, i, P[t, i].x))
            print("On/off status at time t=%d, unit %d: %d" % (t, i, s[t, i].x))
else:
    print("No optimal solution found")

