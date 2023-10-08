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

# Create the model
model = gp.Model("UnitCommitment")

# Create the decision variables
Pg = {}
Sg = {}
Ug = {}
Dg = {}
for t in range(T):
    for g in range(N):
        Pg[g, t] = model.addVar(lb=0, ub=P_max[g], vtype=GRB.CONTINUOUS, name=f"Pg_{g}_{t}")
        Sg[g, t] = model.addVar(vtype=GRB.BINARY, name=f"Sg_{g}_{t}")
        Ug[g, t] = model.addVar(vtype=GRB.BINARY, name=f"Ug_{g}_{t}")
        Dg[g, t] = model.addVar(vtype=GRB.BINARY, name=f"Dg_{g}_{t}")

# Set the objective function
obj = gp.quicksum(C[g] * Pg[g, t] * Sg[g, t] + S[g] * Ug[g, t] + D[g] * Dg[g, t] for g in range(N) for t in range(T))
model.setObjective(obj, GRB.MINIMIZE)

# Add the constraints
# Power balance constraint
for t in range(T):
    model.addConstr(gp.quicksum(Pg[g, t] for g in range(N)) == demand[t], f"Power_Balance_{t}")

# Minimum start-stop time constraint
for g in range(N):
    for t in range(T):
        model.addConstr(gp.quicksum(Ug[g, tau] for tau in range(max(0, t - U_min[g] + 1), t + 1)) <= Sg[g, t], f"Min_Start_{g}_{t}")
        model.addConstr(gp.quicksum(Dg[g, tau] for tau in range(max(0, t - U_max[g] + 1), t + 1)) <= 1 - Sg[g, t], f"Min_Stop_{g}_{t}")

# Power output limit constraint
for g in range(N):
    for t in range(T):
        model.addConstr(Pg[g, t] <= P_max[g] * Sg[g, t], f"Power_Limit_{g}_{t}")
        model.addConstr(Pg[g, t] >= P_min[g] * Sg[g, t], f"Power_Limit_{g}_{t}")

# Ramp power constraint
for g in range(N):
    for t in range(1, T):
        model.addConstr(Pg[g, t - 1] - Pg[g, t] <= RU[g], f"Ramp_Up_{g}_{t}")
        model.addConstr(Pg[g, t] - Pg[g, t - 1] <= RD[g], f"Ramp_Down_{g}_{t}")

# Carbon emission constraint
for t in range(T):
    model.addConstr(gp.quicksum(E[g] * Pg[g, t] for g in range(N)) <= E_max, f"Carbon_Emission_{t}")

# Additional constraints on binary variables
for g in range(N):
    for t in range(1, T):
        model.addConstr(Ug[g, t] >= Sg[g, t] - Sg[g, t - 1], f"Minimum_Start-Up_{g}_{t}")
        model.addConstr(Dg[g, t] >= Sg[g, t - 1] - Sg[g, t], f"Minimum_Shut-Down_{g}_{t}")

# Set Gurobi parameters (optional)
model.setParam("OutputFlag", 1)  # Print solver output

# Optimize the model
model.optimize()

# Check the optimization result
if model.status == GRB.OPTIMAL:
    print("Optimal solution found")
    for t in range(T):
        for g in range(N):
            print(f"Unit {g}, Time {t}: On-Off Status = {Sg[g, t].X}, Power Output = {Pg[g, t].X}")
else:
    print("No solution found")

# Dispose the model
model.dispose()
