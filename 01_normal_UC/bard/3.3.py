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
pi = model.addVars(T, N, vtype=gp.Reals, name="pi")
ui = model.addVars(T, N, vtype=gp.Binary, name="ui")

# Objective function
objective = gp.Minimize(sum((C[i] * pi[t, i] + S[i] * ui[t, i]) for t in range(T) for i in range(N)))

# Constraints
for t in range(T):
    model.addConstr(sum(pi[t, i] for i in range(N)) == demand[t])
    for i in range(N):
        model.addConstr(pi[t, i] <= P_max[i])
        model.addConstr(pi[t, i] >= P_min[i])
        model.addConstr(ui[t, i] - ui[t-1, i] <= 1)
        model.addConstr(pi[t, i] - pi[t-1, i] <= RU[i])
        model.addConstr(pi[t, i] - pi[t-1, i] >= RD[i])
        continue)
        model.addConstr(sum(ui[t, i] for t in range(U_min[i])) >= U_min[i])
        model.addConstr(sum(ui[t, i] for t in range(T - U_max[i] + 1)) <= U_max[i])
model.addConstr(sum(E[i] * pi[t, i] for t in range(T) for i in range(N)) <= E_max)

# Solve the model
model.optimize()

# Print the solution
if model.status == gp.Status.OPTIMAL:
    print("Optimal objective value: ", model.objVal)
    print("Power output:")
    for t in range(T):
        for i in range(N):
            print(pi[t, i].x)
    print("Unit status:")
    for t in range(T):
        for i in range(N):
            print(ui[t, i].x)
else:
    print("No feasible solution found")

