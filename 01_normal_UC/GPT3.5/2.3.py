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

# Create the decision variables
P = {}
u = {}
s = {}
d = {}

for g in range(N):
    for t in range(T):
        P[g, t] = model.addVar(lb=0, ub=P_max[g], vtype=GRB.CONTINUOUS, name=f"P[{g},{t}]")
        u[g, t] = model.addVar(vtype=GRB.BINARY, name=f"u[{g},{t}]")
        s[g, t] = model.addVar(vtype=GRB.BINARY, name=f"s[{g},{t}]")
        d[g, t] = model.addVar(vtype=GRB.BINARY, name=f"d[{g},{t}]")

# Set objective function
obj = (
    gp.quicksum(C[g] * P[g, t] + S[g] * s[g, t] + D[g] * d[g, t]  for g in range(N) for t in range(T))
)
model.setObjective(obj, GRB.MINIMIZE)

# Add constraints
for t in range(T):
    # Power balance constraint
    model.addConstr(gp.quicksum(P[g, t] for g in range(N)) == demand[t], name=f"Power_Balance[{t}]")

    for g in range(N):
        if t > 0:
            # Ramp Power Constraint
            model.addConstr(P[g, t] - P[g, t - 1] <= RU[g] * u[g, t - 1], name=f"Ramp_Up[{g},{t}]")
            model.addConstr(P[g, t - 1] - P[g, t] <= RD[g] * u[g, t - 1], name=f"Ramp_Down[{g},{t}]")

        # Start-up and shut-down constraints
        if t > 0:
            model.addConstr(u[g, t] == u[g, t - 1] + s[g, t] - d[g, t], name=f"Start_Shutdown[{g},{t}]")
        # else:
        #     model.addConstr(u[g, t] == s[g, t] - d[g, t], name=f"Start_Shutdown[{g},{t}]")

        # Start-up and shut-down time constraints
        start_up_sum = gp.quicksum(s[g, tau] for tau in range(max(0, t - U_min[g] + 1), t + 1))
        shut_down_sum = gp.quicksum(d[g, tau] for tau in range(max(0, t - U_max[g] + 1), t + 1))
        model.addConstr(start_up_sum <= u[g, t], name=f"Start_Up_Time[{g},{t}]")
        model.addConstr(shut_down_sum <= 1 - u[g, t], name=f"Shut_Down_Time[{g},{t}]")

        # Power Output Limit Constraint
        model.addConstr(P[g, t] <= P_max[g] * u[g, t], name=f"Power_Output_Limit_Upper[{g},{t}]")
        model.addConstr(P[g, t] >= P_min[g] * u[g, t], name=f"Power_Output_Limit_Lower[{g},{t}]")

# Carbon emission constraint
model.addConstr(gp.quicksum(E[g] * P[g, t] for g in range(N) for t in range(T)) <= E_max, name="Carbon_Emission")

# Optimize the model
model.optimize()

# Print the optimal solution
if model.status == GRB.OPTIMAL:
    print("Optimal Solution:")
    for g in range(N):
        for t in range(T):
            if u[g, t].x > 0.5:
                print(f"Unit {g}: Time {t}, Power output: {P[g, t].x}")

# Dispose of the model
model.dispose()
