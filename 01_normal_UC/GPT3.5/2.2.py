import gurobipy as gp
from gurobipy import GRB

# Define the parameters
T = 10  # total periods
N = 10  # number of units
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

# Create decision variables
P = model.addVars(T, N, lb=0, name="P")
y = model.addVars(T, N, vtype=GRB.BINARY, name="y")
u = model.addVars(T, N, vtype=GRB.BINARY, name="u")
v = model.addVars(T, N, vtype=GRB.BINARY, name="v")

# Set upper bounds for decision variables
for t in range(T):
    for i in range(N):
        P[t, i].ub = P_max[i]

# Set objective function
obj = gp.quicksum(C[i] * P[t, i] + S[i] * u[t, i] + D[i] * v[t, i] for t in range(T) for i in range(N))
model.setObjective(obj, GRB.MINIMIZE)

# Add constraints
power_balance = model.addConstrs((gp.quicksum(P[t, i] for i in range(N)) == demand[t] for t in range(T)), name="power_balance")
power_output_limit = model.addConstrs((P_min[i] * y[t, i] <= P[t, i] for t in range(T) for i in range(N)), name="power_output_limit")
start_up_time = model.addConstrs((gp.quicksum(u[tau, i] for tau in range(t, t + U_min[i])) >= U_min[i] * y[t, i] for t in range(T - U_min[i] + 1) for i in range(N)), name="start_up_time")
shut_down_time = model.addConstrs((gp.quicksum(v[tau, i] for tau in range(t, t + U_max[i])) >= U_max[i] * (1 - y[t, i]) for t in range(T - U_max[i] + 1) for i in range(N)), name="shut_down_time")
ramp_power = model.addConstrs((P[t + 1, i] - P[t, i] <= RU[i] for t in range(T - 1) for i in range(N)), name="ramp_power_up")
ramp_power = model.addConstrs((P[t, i] - P[t + 1, i] <= RD[i] for t in range(T - 1) for i in range(N)), name="ramp_power_down")
carbon_emission = model.addConstrs((C[i] * P[t, i] <= E_max for t in range(T) for i in range(N)), name="carbon_emission")
correlation = model.addConstrs((v[t, i] >= y[t, i] - u[t, i] for t in range(T) for i in range(N)), name="correlation")

# Optimize the model
model.optimize()

# Print optimal solution
if model.status == GRB.OPTIMAL:
    print("Optimal solution found!")
    print(f"Total cost: {model.objVal}")
    for t in range(T):
        print(f"Period {t+1}:")
        for i in range(N):
            if y[t, i].x > 0.5:
                print(f"Unit {i+1} - Power output: {P[t, i].x}")
else:
    print("No solution found.")
