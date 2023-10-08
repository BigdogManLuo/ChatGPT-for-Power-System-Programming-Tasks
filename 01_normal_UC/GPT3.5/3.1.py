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

# Create a new model
model = gp.Model("UnitCommitment")

# Create decision variables
x = model.addVars(N, T, vtype=GRB.BINARY, name="x")  # On-off status
u = model.addVars(N, T, vtype=GRB.BINARY, name="u")  # Start-up status
v = model.addVars(N, T, vtype=GRB.BINARY, name="v")  # Shut-down status
p = model.addVars(N, T, name="p")  # Power output

# Set objective function
model.setObjective(gp.quicksum(C[n] * p[n, t] + S[n] * u[n, t] + D[n] * v[n, t] for n in range(N) for t in range(T)),
                   GRB.MINIMIZE)

# Add constraints
model.addConstrs(gp.quicksum(p[n, t] for n in range(N)) == demand[t] for t in range(T))  # Power balance constraint

model.addConstrs(P_min[n] * x[n, t] <= p[n, t] for n in range(N) for t in range(T))  # Power output limit constraint
model.addConstrs(p[n, t] <= P_max[n] * x[n, t] for n in range(N) for t in range(T))  # Power output limit constraint

model.addConstrs(gp.quicksum(u[n, tau] for tau in range(max(0, t - U_min[n] + 1), t + 1)) <= x[n, t] for n in range(N) for t in range(T))  # Minimum start-up time constraint

model.addConstrs(gp.quicksum(v[n, tau] for tau in range(max(0, t - U_max[n] + 1), t + 1)) <= 1 - x[n, t] for n in range(N) for t in range(T))  # Minimum shut-down time constraint

model.addConstrs(u[n, t] >= x[n, t] - x[n, t - 1] for n in range(N) for t in range(1, T))  # Start-up binary variable constraint
model.addConstrs(v[n, t] >= x[n, t - 1] - x[n, t] for n in range(N) for t in range(1, T))  # Shut-down binary variable constraint

model.addConstrs(p[n, t] - p[n, t - 1] <= RU[n] * x[n, t] for n in range(N) for t in range(1, T))  # Ramp-up constraint
model.addConstrs(p[n, t - 1] - p[n, t] <= RD[n] * x[n, t] for n in range(N) for t in range(1, T))  # Ramp-down constraint

model.addConstrs(gp.quicksum(E[n] * p[n, t] for n in range(N)) <= E_max for t in range(T))  # Carbon emission constraint

# Optimize the model
model.optimize()

# Print the optimal solution
if model.status == GRB.OPTIMAL:
    print("Optimal solution found!")
    print("Total cost: ", model.objVal)
    for n in range(N):
        for t in range(T):
            if x[n, t].x > 0.5:
                print(f"Unit {n+1} is on at time {t+1}, power output: {p[n, t].x}")
else:
    print("No solution found.")

