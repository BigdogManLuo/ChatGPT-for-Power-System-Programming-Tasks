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
model = gp.Model("UnitCommitment")

# Define decision variables
x = {}  # Commitment status (binary)
g = {}  # Generation level
y_startup = {}  # Start-up indicator
y_shutdown = {}  # Shut-down indicator
u = {}  # Minimum up time indicator
d = {}  # Minimum down time indicator

for i in range(N):
    for t in range(T):
        x[i, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{t}")
        g[i, t] = model.addVar(lb=P_min[i], ub=P_max[i], name=f"g_{i}_{t}")
        y_startup[i, t] = model.addVar(vtype=GRB.BINARY, name=f"y_startup_{i}_{t}")
        y_shutdown[i, t] = model.addVar(vtype=GRB.BINARY, name=f"y_shutdown_{i}_{t}")
        u[i, t] = model.addVar(vtype=GRB.BINARY, name=f"u_{i}_{t}")
        d[i, t] = model.addVar(vtype=GRB.BINARY, name=f"d_{i}_{t}")

# Set objective function
model.setObjective(
    gp.quicksum(
        C[i] * (S[i] * y_startup[i, t] + D[i] * y_shutdown[i, t] + g[i, t])
        for i in range(N)
        for t in range(T)
    ),
    GRB.MINIMIZE,
)

# Add constraints

# Power balance constraints
for t in range(T):
    model.addConstr(
        gp.quicksum(g[i, t] for i in range(N)) == demand[t],
        name=f"power_balance_{t}",
    )

# Unit output constraints
for i in range(N):
    for t in range(T):
        model.addConstr(
            g[i, t] >= P_min[i] * x[i, t],
            name=f"min_output_{i}_{t}",
        )
        model.addConstr(
            g[i, t] <= P_max[i] * x[i, t],
            name=f"max_output_{i}_{t}",
        )

# Ramping constraints
for i in range(N):
    for t in range(1, T):
        model.addConstr(
            g[i, t] - g[i, t - 1] <= RU[i],
            name=f"ramp_up_{i}_{t}",
        )
        model.addConstr(
            g[i, t - 1] - g[i, t] <= RD[i],
            name=f"ramp_down_{i}_{t}",
        )

# Binary constraints
for i in range(N):
    for t in range(T):
        model.addConstr(
            x[i, t] >= y_startup[i, t],
            name=f"commitment_startup_{i}_{t}",
        )
        model.addConstr(
            x[i, t] <= 1 - y_shutdown[i, t],
            name=f"commitment_shutdown_{i}_{t}",
        )

# Minimum start-up time constraints
for i in range(N):
    for t in range(T):
        model.addConstr(
            gp.quicksum(u[i, tau] for tau in range(max(t - U_min[i] + 1, 0), t + 1)) >= x[i, t],
            name=f"min_startup_time_{i}_{t}",
        )


# Minimum shut-down time constraints
for i in range(N):
    for t in range(T):
        model.addConstr(
            gp.quicksum(d[i, tau] for tau in range(max(t - U_max[i] + 1, 0), t + 1)) >= 1 - x[i, t],
            name=f"min_shutdown_time_{i}_{t}",
        )


# Carbon emission constraint
model.addConstr(
    gp.quicksum(E[i] * g[i, t] for i in range(N) for t in range(T)) <= E_max,
    name="carbon_emission",
)

# Optimize the model
model.optimize()

# Print optimal solution
if model.status == GRB.OPTIMAL:
    print("Optimal Schedule:")
    for i in range(N):
        print(f"Unit {i + 1}: ", end="")
        for t in range(T):
            if x[i, t].x > 0.5:
                print("ON ", end="")
            else:
                print("OFF ", end="")
        print()
else:
    print("No solution found.")

