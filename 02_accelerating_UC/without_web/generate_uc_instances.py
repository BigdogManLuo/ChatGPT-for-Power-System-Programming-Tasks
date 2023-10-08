from gurobipy import *
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pickle

# Define the parameters
T = 24 # total periods
N = 10  # number of units

#生成样本设置

train_size=400
test_size=100


# Define parameters for the model.
np.random.seed(42)

C=-10*np.random.random(N)+25
S=-100*np.random.random(N)+300
D=-100*np.random.random(N)+200

P_min=-40*np.random.random(N)+50
P_max=-250*np.random.random(N)+500
RU =-200*np.random.random(N)+300
RD=RU
U_min=2*np.ones(N,dtype=np.int32)
U_max=2*np.ones(N,dtype=np.int32)
demand_base = -500*np.random.random(T)+1500

M = 1  # Big M


def generate_train_instances(train_size):
    Demand=[]
    for k in range(train_size):
    
        demand=[num + np.random.randint(-150,150)  for num in demand_base]
        
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
        
        #Record demand 
        Demand.append(demand)
        
        # 写入LP文件
        m.write("data/train/instance"+str(k)+".lp")
        
        #进度显示
        print(str(k)+"/"+str(train_size))
        
    #Write demand to pkl file
    pickle.dump(Demand,open("data/train/demand.pkl","wb"))
        
        
def generate_test_instances(test_size):
    Demand=[]
    Objective=[]
    for k in range(test_size):
    
        demand=[num + np.random.uniform(-50,50)  for num in demand_base]
        
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
        
        
        # Write LP file
        m.write("data/test/instance"+str(k)+".lp")
        
        #Show progress
        print(str(k)+"/"+str(test_size))
        
        #Save demand
        Demand.append(demand)
        #Save Obkective
        m.optimize()
        Objective.append(m.getObjective().getValue())
        
    
    #Save Demand to pkl file
    pickle.dump(Demand,open("data/test/demand.pkl","wb"))
    
    #Save Objective to pkl file
    pickle.dump(Objective,open("data/test/objective.pkl","wb"))
    
        

if __name__ == "__main__":
    generate_train_instances(train_size=train_size)
    generate_test_instances(test_size=test_size)

