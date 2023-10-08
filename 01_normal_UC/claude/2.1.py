import gurobipy as gp
from gurobipy import GRB
# Model parameters
T = 10 
N = 10
C = [19, 18, 19, 20, 16, 19, 24, 18, 11, 23]
S = [290, 276, 210, 296, 299, 243, 260, 255, 222, 227]
D = [101, 120, 147, 101, 192, 125, 112, 179, 171, 191]
P_min = [10, 47, 13, 45, 42, 22, 40, 18, 14, 16]
P_max = [337, 440, 356, 127, 282, 328, 416, 375, 402, 493]
RU = [161, 251, 251, 265, 249, 282, 252, 264, 222, 186]
RD = [161, 251, 251, 265, 249, 282, 252, 264, 222, 186] 
U_min = [2,2,2,2,2,2,2,2,2,2]
U_max = [2,2,2,2,2,2,2,2,2,2]
demand = [801, 1655, 483, 1513, 1742, 1034, 1789, 2375, 1289, 1464]
E = [0.5 + 0.1 * i for i in range(N)]
E_max = 50000
# Create model
m = gp.Model("UC")
# Decision variables
p = m.addVars(N, T, name="Power")
u = m.addVars(N, T, vtype=GRB.BINARY, name="On/Off")
v = m.addVars(N, T, vtype=GRB.BINARY, name="Start")
w = m.addVars(N, T, vtype=GRB.BINARY, name="Stop")
# Objective function
m.setObjective(gp.quicksum(C[i]*p[i,t] + S[i]*v[i,t] + D[i]*w[i,t] for i in range(N) for t in range(T)), GRB.MINIMIZE)
# Constraints
# Power balance
m.addConstrs((gp.quicksum(p[i,t] for i in range(N)) == demand[t] for t in range(T)), name="Demand")
# Min/max limits
m.addConstrs((p[i,t] >= P_min[i]*u[i,t] for i in range(N) for t in range(T)), name="MinLimit") 
m.addConstrs((p[i,t] <= P_max[i]*u[i,t] for i in range(N) for t in range(T)), name="MaxLimit")
# Ramp limits
m.addConstrs((p[i,t] - p[i,t-1] <= RU[i] for i in range(N) for t in range(1,T)), name="RampUp")
m.addConstrs((p[i,t-1] - p[i,t] <= RD[i] for i in range(N) for t in range(1,T)), name="RampDown")
# Min up/down time
m.addConstrs((u[i,t] - u[i,t-1] <= v[i,t] for i in range(N) for t in range(1,T)), name="UpTime")
m.addConstrs((u[i,t-1] - u[i,t] <= w[i,t] for i in range(N) for t in range(1,T)), name="DownTime")
m.addConstrs((gp.quicksum(v[i,t] for t in range(t-U_min[i]+1, t+1)) <= u[i,t]
              for i in range(N) for t in range(U_min[i]-1, T)), name="MinUp")
m.addConstrs((gp.quicksum(w[i,t] for t in range(t-U_max[i]+1, t+1)) <= 1 - u[i,t]
              for i in range(N) for t in range(U_max[i]-1, T)), name="MinDown")
# Emission limit
m.addConstr(gp.quicksum(E[i] * p[i,t] for i in range(N) for t in range(T)) <= E_max, name="EmissionLimit") 

# Optimize
m.optimize() 

# Print solution
print("Total cost: " + str(m.objVal))
for i in range(N):
  for t in range(T):
    if p[i,t].x > 0:
      print("Generator %d produces %g MW in period %d" % (i, p[i,t].x, t))
    if v[i,t].x > 0:
      print("Generator %d starts up in period %d" % (i, t)) 
    if w[i,t].x > 0:
      print("Generator %d shuts down in period %d" % (i, t))