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
m = gp.Model("UnitCommitment")
# Decision variables
P = {}
y = {}
u = {}
d = {}
for i in range(N):
    for t in range(T):
        P[i,t] = m.addVar(lb=0, ub=GRB.INFINITY, name="P_%s_%s"%(i,t))
        y[i,t] = m.addVar(vtype=GRB.BINARY, name="y_%s_%s"%(i,t))
        u[i,t] = m.addVar(vtype=GRB.BINARY, name="u_%s_%s"%(i,t))
        d[i,t] = m.addVar(vtype=GRB.BINARY, name="d_%s_%s"%(i,t))
# Objective function        
m.setObjective(gp.quicksum(C[i]*P[i,t] + S[i]*u[i,t] + D[i]*d[i,t] for i in range(N) for t in range(T)), GRB.MINIMIZE)
# Constraints
for t in range(T):
    # Power balance
    m.addConstr(gp.quicksum(P[i,t] for i in range(N)) == demand[t])
    # Min/max limits
    for i in range(N):
        m.addConstr(P_min[i] * y[i,t] <= P[i,t])
        m.addConstr(P[i,t] <= P_max[i] * y[i,t])
    # Ramp limits
    for i in range(N):
        if t > 0:
            m.addConstr(P[i, t-1] - RD[i] <= P[i,t])
            m.addConstr(P[i,t] <= P[i, t-1] + RU[i])
    # Min up time       
    for i in range(N):
        for t in range(U_min[i]-1, T):
            m.addConstr(gp.quicksum(u[i,j] for j in range(t-U_min[i]+1, t+1)) <= y[i,t])
    # Min down time
    for i in range(N):
        for t in range(U_max[i]-1, T):
            m.addConstr(gp.quicksum(d[i,j] for j in range(t-U_max[i]+1, t+1)) <= 1 - y[i,t])
    # Logic constraints
    for i in range(N):
        for t in range(T):
            if t > 0:
                m.addConstr(y[i,t] - y[i,t-1] == u[i,t] - d[i,t])
                # Carbon emission limit
for t in range(T):
  m.addConstr(gp.quicksum(E[i]*P[i,t] for i in range(N)) <= E_max)
# Optimize
m.optimize()
# Print solution
print('Total cost: %g' % m.objVal)
for i in range(N):
    for t in range(T):
        if P[i,t].x > 0:
            print('Generator %s produces %g MW in period %s' % (i, P[i,t].x, t))
        if y[i,t].x > 0:
            print('Generator %s is on in period %s' % (i, t))
        if u[i,t].x > 0:
            print('Generator %s starts up in period %s' % (i, t))
        if d[i,t].x > 0:
            print('Generator %s shuts down in period %s' % (i, t))