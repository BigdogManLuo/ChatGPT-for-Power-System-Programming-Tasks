from gurobipy import *
import numpy as np
import pickle


def generate_train_instance(train_size):
    cost_coffe=[]

    for k in range(train_size):
    
        #Add random number
        C = [num + np.random.randint(0,2) for num in C_base]
        S = [num + np.random.randint(0,2) for num in S_base]
        D = [num + np.random.randint(0,2) for num in D_base]
        P_min = [num + np.random.randint(0,2) for num in P_min_base]
        P_max = [num + np.random.randint(0,5) for num in P_max_base]
        RU = [num + np.random.randint(0,2) for num in RU_base]
        RD = [num + np.random.randint(0,2) for num in RD_base]
        demand=[num + np.random.randint(-50,50) for num in demand_base]
        
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
        '''
        for i in range(N):
            for t in range(U_min[i], T):
                m.addConstr(sum(V[i][t-j] for j in range(U_min[i])) <= 1, "MinUp_{}_{}".format(i, t))
            for t in range(U_max[i], T):
                m.addConstr(sum(W[i][t-j] for j in range(U_max[i])) <= 1, "MinDown_{}_{}".format(i, t))
        '''
        
        cost_coffe.append(C)
        # Write LP file
        m.write("data/instances/uc/train/instance"+str(k)+".lp")
        
        #Show the progress
        print(str(k)+"/"+str(train_size))
    
    #Write the cost coffe to pkl file
    with open("data/tmp/train/C.pkl","wb") as f:
        pickle.dump(cost_coffe,f)
        
        
        
def generate_test_instance(test_size):
    
    Objs=[]
    cost_coffe=[]
    
    for k in range(test_size):
    
    
        #Add random number
        C = [num + np.random.randint(0,2) for num in C_base]
        S = [num + np.random.randint(0,2) for num in S_base]
        D = [num + np.random.randint(0,2) for num in D_base]
        P_min = [num + np.random.randint(0,2) for num in P_min_base]
        P_max = [num + np.random.randint(0,5) for num in P_max_base]
        RU = [num + np.random.randint(0,2) for num in RU_base]
        RD = [num + np.random.randint(0,2) for num in RD_base]
        demand=[num + np.random.randint(-50,50) for num in demand_base]
        
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
        '''
        for i in range(N):
            for t in range(U_min[i], T):
                m.addConstr(sum(V[i][t-j] for j in range(U_min[i])) <= 1, "MinUp_{}_{}".format(i, t))
            for t in range(U_max[i], T):
                m.addConstr(sum(W[i][t-j] for j in range(U_max[i])) <= 1, "MinDown_{}_{}".format(i, t))
        '''
        
        m.optimize()
        
        
        Objs.append(m.getObjective().getValue())
        cost_coffe.append(C)
        
        # Write LP file
        m.write("data/instances/uc/test/instance"+str(k)+".lp")
        
        #Show the progress
        print(str(k)+"/"+str(test_size))
    
    #Write the Objective Value to pkl file
    with open("data/tmp/test/Objs.pkl","wb") as f:
        pickle.dump(Objs,f)
        
    #Write the cost coffe to pkl file
    with open("data/tmp/test/C.pkl","wb") as f:
        pickle.dump(cost_coffe,f)

if __name__ == "__main__":
    # Define the parameters
    T = 24  # total periods
    N = 10  # number of units

    train_size=500
    test_size=100

    # Define parameters for the model.
    np.random.seed(42)

    C_base=-10*np.random.random(N)+25
    S_base=-100*np.random.random(N)+300
    D_base=-100*np.random.random(N)+200
    P_min_base=-40*np.random.random(N)+50
    P_max_base=-300*np.random.random(N)+500
    RU_base =-200*np.random.random(N)+300
    RD_base=RU_base
    U_min=2*np.ones(N,dtype=np.int32)
    U_max=2*np.ones(N,dtype=np.int32)
    demand_base = -500*np.random.random(T)+1500

    M = 1  # Big M
    
    generate_train_instance(train_size)
    generate_test_instance(test_size)

