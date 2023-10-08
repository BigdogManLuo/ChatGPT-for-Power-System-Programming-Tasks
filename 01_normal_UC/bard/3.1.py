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

# Create the decision variables
u = []
p = []
for t in range(T):
  u.append(model.addVar(vtype=gp.GRB.BINARY, name="u[{}]".format(t)))
  for i in range(N):
    p.append(model.addVar(lb=P_min[i], ub=P_max[i], vtype=gp.GRB.CONTINUOUS, name="p[{}][{}]".format(t, i)))

# Add the objective function
obj = 0
for t in range(T):
  for i in range(N):
    name = "p[{}][{}]".format(t, i)
    model.setVarName(p[t][i], name)
    obj += C[i] * model.getVarByName(name).x + S[i] * model.getVarByName("u[{}]".format(t)).x * (1 - model.getVarByName("u[{}]".format(t - 1)).x) + D[i] * (1 - model.getVarByName("u[{}]".format(t)).x) * model.getVarByName("u[{}]".format(t - 1)).x
model.setObjective(obj, gp.GRB.MINIMIZE)

# Add the constraints
for t in range(T):
  model.addConstr(sum(p[t][i] for i in range(N)) == demand[t])
  for i in range(N):
    model.addConstr(p[t][i] - p[t - 1][i] <= RU[i] * u[t])
    model.addConstr(p[t - 1][i] - p[t][i] <= RD[i] * (1 - u[t]))
    model.addConstr(u[t] - u[t - 1] >= U_min[i])
    model.addConstr(u[t] <= U_max[i])

# Add the carbon emission constraint
model.addConstr(sum(E[i] * p[t][i] for i in range(N) for t in range(T)) <= E_max)

# Optimize the model
model.optimize()

# Print the solution
if model.status == gp.GRB.OPTIMAL:
  print("Total cost: {}".format(model.objVal))
  for t in range(T):
    for i in range(N):
      if u[t].x == 1:
        print("Unit {} is on at time {}".format(i, t))
      print("Power output of unit {} at time {} is {}".format(i, t, p[t][i].x))
else:
  print("No solution found")

