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

# Initialize model
m = Model('UC')

# Create variables
x = m.addVars(N, T, vtype=GRB.BINARY, name="x")
y = m.addVars(N, T, vtype=GRB.BINARY, name="y")
z = m.addVars(N, T, vtype=GRB.BINARY, name="z")
P = m.addVars(N, T, lb=0, vtype=GRB.CONTINUOUS, name="P")

# Set objective
m.setObjective(sum((C[i]*P[i,t] + S[i]*y[i,t] + D[i]*z[i,t]) for i in range(N) for t in range(T)), GRB.MINIMIZE)

# Set constraints
for t in range(T):
    m.addConstr(sum(P[i,t] for i in range(N)) == demand[t], name="demand"+str(t))
    if t > 0:
        for i in range(N):
            m.addConstr(x[i,t] - x[i,t-1] == y[i,t] - z[i,t], name="on_off"+str(i)+"_"+str(t))
            m.addConstr(P[i,t] - P[i,t-1] <= RU[i], name="ramp_up"+str(i)+"_"+str(t))
            m.addConstr(P[i,t-1] - P[i,t] <= RD[i], name="ramp_down"+str(i)+"_"+str(t))
            
for i in range(N):
    for t in range(T):
        m.addConstr(P[i,t] >= P_min[i] * x[i,t], name="min_output"+str(i)+"_"+str(t))
        m.addConstr(P[i,t] <= P_max[i] * x[i,t], name="max_output"+str(i)+"_"+str(t))
        if t < T - U_min[i]:
            m.addConstr(sum(x[i,k] for k in range(t, t + U_min[i])) >= U_min[i] * y[i,t], name="min_up"+str(i)+"_"+str(t))
        if t < T - U_max[i]:
            m.addConstr(sum((1 - x[i,k]) for k in range(t, t + U_max[i])) >= U_max[i] * z[i,t], name="min_down"+str(i)+"_"+str(t))

m.addConstr(sum(E[i]*P[i,t] for i in range(N) for t in range(T)) <= E_max, name="emission")

# Optimize model
m.optimize()
