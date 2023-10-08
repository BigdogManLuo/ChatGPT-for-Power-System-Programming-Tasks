import gurobipy as gp

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

# Create the model
model = gp.Model("unit_commitment")

# Create decision variables
u = model.addVars(T, N, vtype=gp.GRB.BINARY, name="u")
p = model.addVars(T, N, name="p")

# Set objective
model.setObjective(sum(C[i] * p[t, i] + S[i] * u[t, i] + D[i] * (1 - u[t, i]) for t in range(T) for i in range(N)),
                      sense=gp.GRB.MINIMIZE)

# Add constraints
for t in range(T):
    model.addConstr(sum(p[t, i] for i in range(N)) == demand[t])
    for i in range(N):
        model.addConstr(p[t, i] <= P_max[i] * u[t, i])
        model.addConstr(p[t, i] >= P_min[i] * u[t, i])
        if t == 0:
            model.addConstr(p[t, i] <= RU[i] * u[t, i])
        else:
            model.addConstr(p[t, i] - p[t - 1, i] <= RU[i] * u[t, i] - RD[i] * (1 - u[t, i]))
        model.addConstr(u[t, i] - u[t - 1, i] <= 1 - U_min[i])
        model.addConstr(u[t, i] - u[t - 1, i] >= -U_max[i])

# Add carbon emissions constraint
model.addConstr(sum(E[i] * p[t, i] for t in range(T) for i in range(N)) <= E_max)

# Solve the model
model.optimize()

# Print the solution
if model.status == gp.GRB.OPTIMAL:
    print("Optimal objective value: ", model.objVal)
    print("Power output: ", p.values())
    print("Unit commitment status: ", u.values())
else:
    print("The model is not optimal")

