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

# Create the optimization model
model = gp.Model("UnitCommitment")

# Create decision variables
P = model.addVars(N, T, name="P")  # Power output of units
ON = model.addVars(N, T, vtype=GRB.BINARY, name="ON")  # On/off status of units
SU = model.addVars(N, T, vtype=GRB.BINARY, name="SU")  # Start-up status of units
SD = model.addVars(N, T, vtype=GRB.BINARY, name="SD")  # Shut-down status of units

# Set objective function
obj_expr = gp.quicksum(C[g] * P[g, t] + S[g] * SU[g, t] + D[g] * SD[g, t] for g in range(N) for t in range(T))
model.setObjective(obj_expr, GRB.MINIMIZE)

# Add constraints
# Power balance constraint
model.addConstrs((gp.quicksum(P[g, t] for g in range(N)) == demand[t] for t in range(T)), name="PowerBalance")

# Power output limit constraint
model.addConstrs((P[g, t] <= P_max[g] * ON[g, t] for g in range(N) for t in range(T)), name="PowerOutputLimit")
model.addConstrs((P[g, t] >= P_min[g] * ON[g, t] for g in range(N) for t in range(T)), name="MinPowerOutputLimit")

# Minimum start-up and shut-down time constraint
model.addConstrs((SU[g, t] >= ON[g, t] - ON[g, t - 1] for g in range(N) for t in range(1, T)), name="MinStartUp")
model.addConstrs((SD[g, t] >= ON[g, t - 1] - ON[g, t] for g in range(N) for t in range(1, T)), name="MinShutDown")
model.addConstrs((gp.quicksum(SU[g, tau] for tau in range(t - U_min[g] + 1, t + 1)) <= ON[g, t]
                  for g in range(N) for t in range(U_min[g], T)), name="MinUpTime")
model.addConstrs((gp.quicksum(SD[g, tau] for tau in range(t - U_max[g] + 1, t + 1)) <= 1 - ON[g, t]
                  for g in range(N) for t in range(U_max[g], T)), name="MinDownTime")

# Ramp power constraint
model.addConstrs((P[g, t] - P[g, t - 1] <= RU[g] for g in range(N) for t in range(1, T)), name="RampUp")
model.addConstrs((P[g, t - 1] - P[g, t] <= RD[g] for g in range(N) for t in range(1, T)), name="RampDown")

# Carbon emission constraint
model.addConstr(gp.quicksum(E[g] * P[g, t] for g in range(N) for t in range(T)) <= E_max, name="CarbonEmission")

# Set Gurobi parameters
model.setParam("OutputFlag", 0)  # Disable Gurobi output

# Optimize the model
model.optimize()

# Check optimization status
if model.status == GRB.OPTIMAL:
    print("Optimal solution found")
else:
    print("No feasible solution found")

# Print optimal unit commitment schedule
for t in range(T):
    print("Time period", t)
    print("Unit\tStatus\tPower Output")
    for g in range(N):
        print(g, "\t", int(ON[g, t].x), "\t", int(P[g, t].x))
    print()
