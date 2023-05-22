from gurobipy import *
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Define the parameters
T = 10  # total periods
N = 10  # number of units


# Define parameters for the model.
C = [19, 18, 19, 26, 20, 27, 27, 18, 11, 27]  # Cost coefficients
S = [290, 276, 210, 296, 299, 243, 260, 255, 222, 227]  # Start-up costs
D = [101, 120, 147, 101, 192, 125, 112, 179, 171, 191] # Shutdown costs
P_min = [10, 47, 13, 45, 42, 22, 40, 18, 14, 16]  # Min power output
P_max = [337, 440, 356, 127, 282, 328, 416, 375, 402, 493]  # Max power output
RU = [161, 251, 251, 265, 249, 282, 252, 264, 222, 186] # Ramp-up rate
RD = [161, 251, 251, 265, 249, 282, 252, 264, 222, 186]  # Ramp-down rate
U_min = [2,2,2,2,2,2,2,2,2,2] # Minimum up time
U_max = [2,2,2,2,2,2,2,2,2,2]   # Minimum down time
#demand = [801, 1655, 483, 1513, 1742, 1034, 1789, 2375, 1289, 1464,1324, 601, 1036, 2263, 1061, 2414, 358, 1918, 2165, 2281,1559, 2030, 1722, 1322]  # Load demand
demand = [801, 1655, 483, 1513, 1742, 1034, 1789, 2375, 1289, 1464]
M = 10  # Big M

# Create a new model
m = Model("UC")

# Create variables
P = [[m.addVar(lb=0, ub=P_max[i], name="P_{}_{}".format(i, t)) for t in range(T)] for i in range(N)]  # Power output
U = [[m.addVar(vtype=GRB.BINARY, name="U_{}_{}".format(i, t)) for t in range(T)] for i in range(N)]  # Unit status
V = [[m.addVar(vtype=GRB.BINARY, name="V_{}_{}".format(i, t)) for t in range(T)] for i in range(N)]  # Start-up status
W = [[m.addVar(vtype=GRB.BINARY, name="W_{}_{}".format(i, t)) for t in range(T)] for i in range(N)]  # Shutdown status

# Set objective
m.setObjective(sum(C[i]*P[i][t] + S[i]*V[i][t] + D[i]*W[i][t] for i in range(N) for t in range(T)), GRB.MINIMIZE)

# Add constraints
for t in range(T):
    # Power balance constraint
    m.addConstr(sum(P[i][t] for i in range(N)) == demand[t], "PowerBalance_{}".format(t))
    
    for i in range(N):
        # Unit status constraint
        m.addConstr(P_min[i]*U[i][t] <= P[i][t], "MinPower_{}_{}".format(i, t))
        m.addConstr(P[i][t] <= P_max[i]*U[i][t], "MaxPower_{}_{}".format(i, t))

        # Startup and shutdown constraints
        if t > 0:
            m.addConstr(V[i][t] >= U[i][t] - U[i][t-1], "Startup1_{}_{}".format(i, t))
            m.addConstr(V[i][t] <= U[i][t] - U[i][t-1] + M*(1 - U[i][t-1]), "Startup2_{}_{}".format(i, t))

            m.addConstr(W[i][t] >= U[i][t-1] - U[i][t], "Shutdown1_{}_{}".format(i, t))
            m.addConstr(W[i][t] <= U[i][t-1] - U[i][t] + M*(1 - U[i][t]), "Shutdown2_{}_{}".format(i, t))

            # Ramp constraints
            m.addConstr(P[i][t] - P[i][t-1] <= RU[i], "RampUp_{}_{}".format(i, t))
            m.addConstr(P[i][t-1] - P[i][t] <= RD[i], "RampDown_{}_{}".format(i, t))

# Add minimum up/down time constraints

for i in range(N):
    for t in range(U_min[i], T):
        m.addConstr(sum(V[i][t-j] for j in range(U_min[i])) <= 1, "MinUp_{}_{}".format(i, t))
    for t in range(U_max[i], T):
        m.addConstr(sum(W[i][t-j] for j in range(U_max[i])) <= 1, "MinDown_{}_{}".format(i, t))



# Optimize model
m.optimize()


# Extract the optimal power output values
P_val = [[P[i][t].X for t in range(T)] for i in range(N)]

# Create a 3D bar plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

_x = np.arange(T)
_y = np.arange(N)
_xx, _yy = np.meshgrid(_x, _y)
x, y = _xx.ravel(), _yy.ravel()

top = np.array(P_val).ravel()
bottom = np.zeros_like(top)

# Adjust the width and depth of each bar to make them square
width=0.8
depth=0.8

# Add edges and color gradient
colors = plt.cm.viridis(top / max(top))  # Normalize the colors

ax.bar3d(x, y, bottom, width, depth, top, color=colors, shade=True, edgecolor='black')
ax.set_xlabel('Time')
ax.set_ylabel('Unit')
ax.set_zlabel('Power Output')

# Adjust the azimuth angle
ax.view_init(azim=-36,elev=14) # Adjust this value to change the azimuth angle

# Add a color bar
cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax, orientation='vertical', fraction=0.03, pad=0.05)
cbar.set_label('Power Output')

plt.savefig("UC_Result.png",dpi=1000)

plt.show()