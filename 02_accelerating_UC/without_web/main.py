import os
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB
import torch.utils.data as data
import pickle
import numpy as np
import time

np.random.seed(42)
N=10
C=-10*np.random.random(N)+25
S=-100*np.random.random(N)+300
D=-100*np.random.random(N)+200

def extract_features_and_labels(file_path):
    # Read the LP file using gurobipy
    model = gp.read(file_path)
    
    model.optimize() #Systhis
    
    if not (model.status == GRB.Status.OPTIMAL):
        print(file_path)
    
    labels = defaultdict(float)
    
    # Extract features and labels. Here, I'm making broad assumptions.
    # Adjust these based on the actual content of your LP files.
    
    '''
    # Variables might be treated as features. We capture their lower and upper bounds.
    for var in model.getVars():
        var_name = var.VarName
        features[var_name] = (var.LB, var.UB)
    '''
    
    #Extract features from data/train/demand.pkl
    #features=pickle.load(open("data/train/demand/demand.pkl","rb"))
    
    # Extend labels to capture unit outputs
    for var in model.getVars():
        
        var_name = var.VarName
        labels[var_name + "_output"] = var.X  # X represents the value of the variable in Gurobi

    # Add total dispatching cost (assuming minimization problem)
    #labels["total_cost"] = model.getObjective().getValue() 
    
    return  labels

def process_lp_files(directory_path):
    #Load features from pkl file
    feature_path=directory_path+str("demand.pkl")
    features_batch = pickle.load(open(feature_path,"rb"))
    labels_batch = []
    
    for filename in os.listdir(directory_path):
        if filename.endswith(".lp"):
            file_path = os.path.join(directory_path, filename)
            labels = extract_features_and_labels(file_path)
            labels_batch.append(labels)
    
    return features_batch, labels_batch


def dicts_to_tensors(data_batch):
    # Convert a batch of dictionaries to tensors.
    # This assumes all dictionaries in the batch have the same keys and order.
    keys = list(data_batch[0].keys())
    tensor_data = torch.tensor([[d[k] for k in keys] for d in data_batch])
    return tensor_data


# ------------------ Neural Network Design ------------------

class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_units):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)# 1 for dispatch cost + number of units
        self.fc3 = nn.Linear(hidden_size, hidden_size)# 1 for dispatch cost + number of units
        self.fc4 = nn.Linear(hidden_size, num_units)# 1 for dispatch cost + number of units        self.relu = nn.ReLU()
        self.relu= nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        return x


# Training function
def train_model(model, train_loader, loss_fn, optimizer, num_epochs=10):
    model.train()  # Set the model to training mode
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        
        for batch_idx, (features, labels) in enumerate(train_loader):
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(features)
            loss = loss_fn(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")


# Validation function
def validate_model(model, val_loader, loss_fn):
    model.eval()  # Set the model to evaluation mode
    total_val_loss = 0.0
    with torch.no_grad():
        for features, labels in val_loader:
            outputs = model(features)
            loss = loss_fn(outputs, labels)
            total_val_loss += loss.item()
    return total_val_loss/len(val_loader)

# ------------------ TEST Model ------------------

def checkContraints(P,V,W,demand):
    status="feasible"
    for t in range(24):
        if sum(P[i][t] for i in range(10))<=1.21*demand[t] and sum(P[i][t] for i in range(10))>=0.79*demand[t]:
            continue
        else:
            status="infeasible"
            break
    
    return status,sum(P[i][t]*C[i] for i in range(10) for t in range(24))

def getOriginObjValue(ins_index):
    m = gp.read("data/test/instance"+str(ins_index)+".lp")
    m.optimize()
    return m.getObjective().getValue()


def test_model(dataset):
    model.eval()  # Set the model to evaluation mode
    
    #Load Demand from pkl file
    with open('data/test/demand.pkl', 'rb') as f:
        Demand = pickle.load(f)
    #Load Objective from pkl file
    with open('data/test/Objective.pkl', 'rb') as f:
        Objective = pickle.load(f)
    
    total_test_loss = 0.0
    ins_idx=0
    Objs_true=[]
    Objs_pred=[]
    Status=[]
    with torch.no_grad():
        for features, labels in dataset:
            #Get demand
            demand=Demand[ins_idx]
            #Forward pass
            outputs = model(features)
            #Inverse Normalization
            outputs=outputs*Y_std.numpy()+Y_mean.numpy()
            #Convert outputs to numpy array
            outputs=outputs.numpy()
            #Extract P,U,V,W
            P=outputs[0][0:240].reshape(10,24)
            U=outputs[0][240:480].reshape(10,24)
            V=outputs[0][480:720].reshape(10,24)
            W=outputs[0][720:960].reshape(10,24)
                    
            #Check the Constraints if all constraints are satisfied,then status="feasible", otherwise status="infeasible"
            status,dispatch_cost=checkContraints(P,V,W,demand)
            #Get the original objective value
            #origin_obj_value=getOriginObjValue(ins_idx)
            origin_obj_value=Objective[ins_idx]
            #Calculate the gap
            #gap=(dispatch_cost-origin_obj_value)/origin_obj_value
            #Record
            Objs_true.append(origin_obj_value)
            Objs_pred.append(dispatch_cost)
            Status.append(status)
            
            #Update ins_idx
            ins_idx+=1
            
    return Objs_true,Objs_pred,Status



#%% Visualization
import matplotlib.pyplot as plt

# Scatter Plot Objs_true vs Objs_pred with different colors for different status
# if status is "feasible", then the color is blue, otherwise the color is red
def scatter_plot(Objs_true,Objs_pred,Status):
    Objs_true=np.array(Objs_true)
    Objs_pred=np.array(Objs_pred)

    #Normalize Objs_true and Objs_pred with max-min normalization
    Objs_pred=(Objs_pred-Objs_true.min())/(Objs_true.max()-Objs_true.min())
    Objs_true=(Objs_true-Objs_true.min())/(Objs_true.max()-Objs_true.min())
    
    #Objs_true_mean=Objs_true.mean()
    #Objs_true_std=Objs_true.std()
    #Objs_true=(Objs_true-Objs_true_mean)/Objs_true_std
    #Objs_true=Objs_true/Objs_true.max()
    
    #Objs_pred_mean=Objs_pred.mean()
    #Objs_pred_std=Objs_pred.std()
    #Objs_pred=(Objs_pred-Objs_true_mean)/Objs_true_std
    #Objs_pred=Objs_pred/Objs_pred.max()
    
    
    #Calculate R^2 score
    R2_score=1-np.sum((Objs_true-Objs_pred)**2)/np.sum((Objs_true-Objs_true.mean())**2)

    
    plt.figure(figsize=(10,6))
    for i in range(len(Objs_true)):
        if Status[i]=="feasible":
            plt.scatter(Objs_true[i],Objs_pred[i],c="blue")
        else:
            plt.scatter(Objs_true[i],Objs_pred[i],c="red",marker="x")
    plt.xlabel("Normalized Cost(true)",fontsize=20,fontname = 'Arial')
    plt.ylabel("Normalized Cost(pred)",fontsize=20,fontname = 'Arial')
    #Plot a dividing line
    plt.plot([min(min(Objs_pred),min(Objs_true)), max(max(Objs_pred),max(Objs_true))],
             [min(min(Objs_pred),min(Objs_true)), max(max(Objs_pred),max(Objs_true))], 'k-', lw=1)
    
    #Show R2_score in title
    plt.title(r'$R^2$'+'='+str(round(R2_score,4)),fontsize=20,fontname = 'Arial')
    

    #Change xlim and ylim to see the details
    plt.xlim(min(min(Objs_pred),min(Objs_true))-1,  max(max(Objs_pred),max(Objs_true))+1)
    plt.ylim(min(min(Objs_pred),min(Objs_true))-1,  max(max(Objs_pred),max(Objs_true))+1)
    
    #Legend  if status is "feasible", then the color is blue, otherwise the color is red
    blue_patch = plt.plot([],[], 'o', color='blue', label='Feasible')
    red_patch = plt.plot([],[], 'x', color='red', label='Infeasible')
    plt.legend(handles=[blue_patch[0],red_patch[0]],loc='upper left')
    
    #Set legend fontname to Times New Roman
    plt.legend(prop={'family':'Arial','size':23})
    
    #Set xticks and yticks fontname to Times New Roman
    plt.xticks(fontname = 'Arial',fontsize=20)
    plt.yticks(fontname = 'Arial',fontsize=20)
    
    ax=plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    
    #Save the figure with dpi=660
    plt.savefig("scatter_plot.png",dpi=660)
    plt.show()

    

if __name__ == "__main__":

    # Specify the directory path
    directory_path = "data/train/"
    features_batch, labels_batch = process_lp_files(directory_path)
    features_batch=np.array(features_batch,dtype=np.float32)
    
    # Convert features to tensors named X
    X= torch.from_numpy(features_batch)
    Y = dicts_to_tensors(labels_batch)
    
    
    # Normalize the X and Y with zero mean and standard deviation
    X_mean = X.mean()
    X_std = X.std()
    X=(X-X_mean) / (X_std +1e-7)
    Y_mean = Y.mean()
    Y_std = Y.std()
    Y=(Y-Y_mean) / (Y_std +1e-7)
    
    
    # Split data into training and validation sets
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    Y_train, Y_val = Y[:train_size], Y[train_size:]

    input_size = X_train.size(1)
    hidden_size = 2048  # This can be adjusted
    output_size = Y_train.size(1)

    model = FeedForwardNN(input_size, hidden_size, output_size)

    # Define a loss function and optimizer
    loss_fn = nn.MSELoss()  # Mean squared error, can be changed based on the problem nature
    optimizer = optim.Adam(model.parameters(), lr=0.021)  # Learning rate can be adjusted

    # ------------------ Training ------------------
    # Convert data to PyTorch datasets and loaders
    batch_size = 64  # Adjust as needed

    train_dataset = data.TensorDataset(X_train, Y_train)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    #val_dataset = data.TensorDataset(X_val, Y_val)
    #val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Train the model
    num_epochs = 30  # Adjust as needed
    train_model(model, train_loader, loss_fn, optimizer, num_epochs)

    # Validate the model
    #val_loss = validate_model(model, val_loader, loss_fn)
    #print(f"Validation Loss: {val_loss:.4f}")
    
    # ------------------ Test ------------------
    
    #Load Testset
    directory_path = "data/test/"
    features_batch, labels_batch = process_lp_files(directory_path)
    features_batch=np.array(features_batch,dtype=np.float32)
    
    # Convert features and labels to tensors
    X_test= torch.from_numpy(features_batch)
    Y_test= dicts_to_tensors(labels_batch)
    
    #Normalize X_test and Y_test with the same mean and std as X_train and Y_train
    X_test=(X_test-X_mean) / (X_std +1e-7)
    Y_test=(Y_test-Y_mean) / (Y_std +1e-7)
    
    # Convert data to PyTorch datasets and loaders
    dataset= data.TensorDataset(X_test, Y_test)
    test_loader = data.DataLoader(dataset, batch_size=1, shuffle=False)
    Objs_true,Objs_pred,Status=test_model(test_loader)
    
    #Scatter Plot
    scatter_plot(Objs_true,Objs_pred,Status)
    
    '''
    #Save Objs_true,Objs_pred,Status into savedata/... pkl format
    with open('savedata/Objs_true.pkl', 'wb') as f:
        pickle.dump(Objs_true, f)
    with open('savedata/Objs_pred.pkl', 'wb') as f:
        pickle.dump(Objs_pred, f)
    with open('savedata/Status.pkl', 'wb') as f:
        pickle.dump(Status, f)
    '''
    
    # Load Objs_true,Objs_pred,Status from savedata/... pkl format
    '''
    with open('savedata/Objs_true.pkl', 'rb') as f:
        Objs_true = pickle.load(f)
    with open('savedata/Objs_pred.pkl', 'rb') as f:
        Objs_pred = pickle.load(f)
    with open('savedata/Status.pkl', 'rb') as f:
        Status = pickle.load(f)
    ''' 
    
    
    


    





















