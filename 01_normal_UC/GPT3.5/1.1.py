import gurobipy as gp
from gurobipy import GRB

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
U_min = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]  # Minimum up time
U_max = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]  # Minimum down time
demand = [801, 1655, 483, 1513, 1742, 1034, 1789, 2375, 1289, 1464]  # Load demand
E = [0.5 + 0.1 * i for i in range(N)]  # Carbon emissions rate
E_max = 50000  # Maximum total carbon emissions

# Create a Gurobi model
model = gp.Model("Unit_Commitment")

# Define decision variables
P = {}  # Power output
X = {}  # Startup
Y = {}  # Shutdown
I = {}  # Status (on/off)

for i in range(N):
    for t in range(T):
        P[i, t] = model.addVar(vtype=GRB.CONTINUOUS, name=f"P_{i}_{t}")
        X[i, t] = model.addVar(vtype=GRB.BINARY, name=f"X_{i}_{t}")
        Y[i, t] = model.addVar(vtype=GRB.BINARY, name=f"Y_{i}_{t}")
        I[i, t] = model.addVar(vtype=GRB.BINARY, name=f"I_{i}_{t}")

# Set objective function
model.setObjective(
    gp.quicksum(C[i] * P[i, t] + S[i] * X[i, t] + D[i] * Y[i, t]  for i in range(N) for t in range(T)),
    sense=GRB.MINIMIZE,
)

# Add constraints
# Power balance constraint
for t in range(T):
    model.addConstr(gp.quicksum(P[i, t] for i in range(N)) == demand[t], name=f"Power_Balance_{t}")

# Minimum up-time and down-time constraints
for i in range(N):
    for t in range(T - U_min[i] + 1):
        model.addConstr(gp.quicksum(I[i, tau] for tau in range(t, t + U_min[i])) >= U_min[i] * X[i, t], name=f"Min_Up_{i}_{t}")
    for t in range(T - U_max[i] + 1):
        model.addConstr(gp.quicksum(1 - I[i, tau] for tau in range(t, t + U_max[i])) >= U_max[i] * Y[i, t], name=f"Min_Down_{i}_{t}")

# Ramping constraints
for i in range(N):
    for t in range(1, T):
        model.addConstr(P[i, t] - P[i, t - 1] <= RU[i], name=f"Ramp_Up_{i}_{t}")
        model.addConstr(P[i, t - 1] - P[i, t] <= RD[i], name=f"Ramp_Down_{i}_{t}")

# Startup and shutdown constraints
for i in range(N):
    for t in range(1, T):
        model.addConstr(X[i, t] - Y[i, t] == I[i, t] - I[i, t - 1], name=f"Startup_Shutdown_{i}_{t}")

# Minimum and maximum power output limits
for i in range(N):
    for t in range(T):
        model.addConstr(P[i, t] >= P_min[i] * I[i, t], name=f"Min_Power_{i}_{t}")
        model.addConstr(P[i, t] <= P_max[i] * I[i, t], name=f"Max_Power_{i}_{t}")

# Carbon emission constraint
model.addConstr(gp.quicksum(E[i] * P[i, t] for i in range(N) for t in range(T)) <= E_max, name="Carbon_Emission_Limit")

# Optimize the model
model.optimize()

# Print the results
if model.status == GRB.OPTIMAL:
    print("Optimal solution found!")
    for i in range(N):
        for t in range(T):
            if I[i, t].x > 0.5:
                print(f"Unit {i} is ON at time {t}, P_{i}_{t} = {P[i, t].x}")
else:
    print("No optimal solution found.")

# Dispose of the model
model.dispose()
