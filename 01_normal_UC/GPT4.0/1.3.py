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

# Add decision variables
X = m.addVars(N, T, vtype=GRB.BINARY, name="X")  # on/off status
P = m.addVars(((i, t) for i in range(N) for t in range(T)), lb=0, ub={i:P_max[i] for i in range(N)}, name="P")  # power output
Z = m.addVars(N, T, vtype=GRB.BINARY, name="Z")  # start-up status
V = m.addVars(N, T, vtype=GRB.BINARY, name="V")  # shutdown status

# Set objective
m.setObjective(quicksum(C[i]*P[i,t] + S[i]*Z[i,t] + D[i]*V[i,t] for i in range(N) for t in range(T)), GRB.MINIMIZE)

# Add constraints
for t in range(T):
    m.addConstr(quicksum(P[i, t] for i in range(N)) == demand[t], "PowerBalance%d" % t)

for i in range(N):
    for t in range(T):
        m.addConstr(P[i, t] <= P_max[i]*X[i, t], "MaxPower%d_%d" % (i, t))
        m.addConstr(P[i, t] >= P_min[i]*X[i, t], "MinPower%d_%d" % (i, t))

        if t > 0:
            m.addConstr(P[i, t] - P[i, t-1] <= RU[i], "RampUp%d_%d" % (i, t))
            m.addConstr(P[i, t-1] - P[i, t] <= RD[i], "RampDown%d_%d" % (i, t))

for i in range(N):
    m.addConstr(quicksum(E[i]*P[i,t] for t in range(T)) <= E_max, "MaxEmission%d" % i)

# Add start-up and shut-down constraints
for i in range(N):
    for t in range(1, T):
        m.addConstr(Z[i,t] >= X[i,t] - X[i,t-1], "StartUp%d_%d" % (i, t))
        m.addConstr(V[i,t] >= X[i,t-1] - X[i,t], "ShutDown%d_%d" % (i, t))

# Add minimum up and down time constraints
for i in range(N):
    for t in range(T):
        if t < U_min[i]:
            m.addConstr(quicksum(X[i,t+k] for k in range(U_min[i])) >= U_min[i]*Z[i,t], "MinUp%d_%d" % (i, t))
        if t < U_max[i]:
            m.addConstr(quicksum(1 - X[i,t+k] for k in range(U_max[i])) >= U_max[i]*V[i,t], "MinDown%d_%d" % (i, t))

# Solve the model
m.optimize()

# Print the optimal solution
if m.status == GRB.OPTIMAL:
    for v in m.getVars():
        print('%s: %g' % (v.varName, v.x))
