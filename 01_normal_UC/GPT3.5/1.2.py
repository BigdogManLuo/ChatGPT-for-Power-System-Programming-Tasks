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

# Create a new Gurobi model
model = gp.Model("Unit_Commitment")

# Define decision variables
u = {}  # On/Off status of units
v = {}  # Start-up of units
w = {}  # Shutdown of units
for i in range(N):
    for t in range(T):
        u[i, t] = model.addVar(vtype=GRB.BINARY, name=f"u_{i}_{t}")
        v[i, t] = model.addVar(vtype=GRB.BINARY, name=f"v_{i}_{t}")
        w[i, t] = model.addVar(vtype=GRB.BINARY, name=f"w_{i}_{t}")

# Set objective function: minimize total operating cost
model.setObjective(
    gp.quicksum(
        (C[i] * u[i, t] + S[i] * v[i, t] + D[i] * w[i, t]) for i in range(N) for t in range(T)
    ),
    GRB.MINIMIZE,
)

# Add constraints
# Power balance constraint
for t in range(T):
    model.addConstr(
        gp.quicksum(P_min[i] * u[i, t] for i in range(N))
        <= gp.quicksum(P_max[i] * u[i, t] for i in range(N))
    )
    model.addConstr(
        gp.quicksum(P_min[i] * u[i, t] for i in range(N))
        <= demand[t]
    )
    model.addConstr(
        demand[t]
        <= gp.quicksum(P_max[i] * u[i, t] for i in range(N))
    )

# Ramp-up and ramp-down constraints
for i in range(N):
    for t in range(1, T):
        model.addConstr(
            -RU[i] <= gp.quicksum(P_min[j] * u[j, t] for j in range(i + 1)) - gp.quicksum(P_min[j] * u[j, t - 1] for j in range(i + 1))
        )
        model.addConstr(
            gp.quicksum(P_min[j] * u[j, t] for j in range(i + 1)) - gp.quicksum(P_min[j] * u[j, t - 1] for j in range(i + 1))
            <= RU[i]
        )
        model.addConstr(
            -RD[i] <= gp.quicksum(P_min[j] * u[j, t - 1] for j in range(i + 1)) - gp.quicksum(P_min[j] * u[j, t] for j in range(i + 1))
        )
        model.addConstr(
            gp.quicksum(P_min[j] * u[j, t - 1] for j in range(i + 1)) - gp.quicksum(P_min[j] * u[j, t] for j in range(i + 1))
            <= RD[i]
        )

# Minimum start-up and shut-down time constraints
for i in range(N):
    for t in range(1, T):
        for tau in range(1, min(t + 1, S[i] + 1)):
            model.addConstr(
                v[i, t] >= u[i, t - tau]
            )

for i in range(N):
    for t in range(1, T):
        for tau in range(1, min(t + 1, D[i] + 1)):
            model.addConstr(
                w[i, t] >= u[i, t] - u[i, t - tau]
            )

# Carbon emission constraint
for t in range(T):
    model.addConstr(
        gp.quicksum(E[i] * P_min[i] * u[i, t] for i in range(N))
        <= E_max
    )

# Minimum up and down time constraints
for i in range(N):
    for t in range(1, T):
        for tau in range(1, min(t + 1, U_min[i] + 1)):
            model.addConstr(
                u[i, t] - u[i, t - tau] <= 0
            )

for i in range(N):
    for t in range(1, T):
        for tau in range(1, min(t + 1, U_max[i] + 1)):
            model.addConstr(
                u[i, t] - u[i, t - tau] >= 0
            )

# Optimize the model
model.optimize()

# Print the results
if model.status == GRB.OPTIMAL:
    print("Optimal Solution Found!")
    for i in range(N):
        for t in range(T):
            if u[i, t].x > 0:
                print(f"Unit {i} is on at time period {t}")
            if v[i, t].x > 0:
                print(f"Unit {i} starts up at time period {t}")
            if w[i, t].x > 0:
                print(f"Unit {i} shuts down at time period {t}")
else:
    print("No optimal solution found.")

# Print the total cost
print(f"Total Cost: ${model.objVal:.2f}")
