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
Pg = model.addVars(N, T, lb=0, name="Pg")  # Power output
U_on = model.addVars(N, T, vtype=GRB.BINARY, name="U_on")  # On/off status
U_start = model.addVars(N, T, vtype=GRB.BINARY, name="U_start")  # Start-up status
U_stop = model.addVars(N, T, vtype=GRB.BINARY, name="U_stop")  # Shut-down status

# Set upper bound for Pg variables
for g in range(N):
    for t in range(T):
        Pg[g, t].ub = P_max[g]


# Set objective function
obj = gp.quicksum(C[g] * Pg[g, t] + S[g] * U_start[g, t] + D[g] * U_stop[g, t] 
                  for g in range(N) for t in range(T))
model.setObjective(obj, GRB.MINIMIZE)

# Add constraints
# Power balance constraint
power_balance = model.addConstrs(gp.quicksum(Pg[g, t] for g in range(N)) == demand[t] for t in range(T))

# Power output limit constraint
power_limit = model.addConstrs((Pg[g, t] >= P_min[g] * U_on[g, t]
                                for g in range(N) for t in range(T)))
power_limit = model.addConstrs((Pg[g, t] <= P_max[g] * U_on[g, t]
                                for g in range(N) for t in range(T)))


# Minimum start-stop time constraint
start_stop_time = model.addConstrs((U_start[g, t] + U_stop[g, t] <= U_on[g, t]
                                    for g in range(N) for t in range(T)))
min_up_time = model.addConstrs((gp.quicksum(U_start[g, tau] for tau in range(max(1, t - U_min[g] + 1), t + 1))
                                <= U_on[g, t] for g in range(N) for t in range(T)))
min_down_time = model.addConstrs((gp.quicksum(U_stop[g, tau] for tau in range(max(1, t - U_min[g] + 1), t + 1))
                                  <= 1 - U_on[g, t] for g in range(N) for t in range(T)))

# Ramp power constraint
ramp_up = model.addConstrs((Pg[g, t] - Pg[g, t - 1] <= RU[g] * U_on[g, t]
                            for g in range(N) for t in range(1, T)))
ramp_down = model.addConstrs((Pg[g, t - 1] - Pg[g, t] <= RD[g] * U_on[g, t]
                              for g in range(N) for t in range(1, T)))

# Carbon emission constraint
carbon_emission = model.addConstrs((gp.quicksum(E[g] * Pg[g, t] for g in range(N)) <= E_max
                                    for t in range(T)))

# Correlation Constraint
correlation = model.addConstrs((U_on[g, t] - U_on[g, t - 1] <= U_start[g, t]
                                for g in range(N) for t in range(1, T)))
correlation = model.addConstrs((U_on[g, t - 1] - U_on[g, t] <= U_stop[g, t]
                                for g in range(N) for t in range(1, T)))

# Optimize the model
model.optimize()

# Check optimization status
if model.status == GRB.OPTIMAL:
    print("Optimal solution found")

    # Print the optimal schedule and power output
    for t in range(T):
        print("Time period:", t)
        for g in range(N):
            if U_on[g, t].x > 0.5:
                print("Unit", g, "is ON with power output:", Pg[g, t].x)
            else:
                print("Unit", g, "is OFF")
        print()
else:
    print("No solution found")

# Retrieve the objective value
total_cost = model.objVal
print("Total cost:", total_cost)
