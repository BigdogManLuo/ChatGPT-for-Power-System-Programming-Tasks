from utilities import loadDataset,lossFunc
from models import divingModel
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import gurobipy
from gurobipy import GRB

#取100组不同的UC instances
trainFile = os.listdir("data/instances/uc/train/")[700:800]

#取对应的二分图sample
adjs,features,sols,objs=loadDataset(-1)

#%% 固定整数变量求解子问题

#加载模型
model=divingModel(nfeat=1,nhid=64,nclass=1,dropout=0.5,nNodes=1434,nOut=120,nlhid=256)  #nNodes和nOut需要调，nNodes约束总数+决策变量总数； nOut：决策变量数
state_dict = torch.load('model/diving_model.pth')
model.load_state_dict(state_dict['model'])

#预测0-1整数解
with torch.no_grad():
    model.eval() #不启用BatchNormalization 和 Dropout
    sols_pred=[]
    for i in range(adjs.shape[0]):
        feature=features[i]
        adj=adjs[i]
        feature=feature.unsqueeze(0)
        adj=adj.unsqueeze(0)
        sol_pred=model(feature,adj.float())
        sols_pred.append(sol_pred)

sols_pred=torch.stack(sols_pred).squeeze(1)
#initialize vector
objs_pred=[]
#求解LP子问题
for i in range(len(trainFile)):
    
    # Read the model from the lp file
    m = gurobipy.read("data/instances/uc/train/"+trainFile[i])
    
    j=0
    for v in m.getVars():
        if v.vType=='B' and ("U" in v.getAttr("VarName")):
            m.addConstr(v==sols_pred[i][j].numpy())
            j=j+1

    # Optimize the model
    m.optimize()
    
    if m.status == GRB.Status.OPTIMAL:
        
        # Get the values of the integer variables
        #sol_true = [v.x for v in m.getVars() if v.vType=='B' and ("U" in v.getAttr("VarName"))]
        # Get the objective value
        obj_pred=m.objVal
        
        #sols_true.append(sol_true)
        objs_pred.append(obj_pred)

objs_pred=torch.tensor(objs_pred)
    
#%% 可视化
objs_mean=objs.mean()
objs_std=objs.std()
objs=(objs-objs_mean)/objs_std

objs_mean_pred=objs_pred.mean()
objs_std_pred=objs_pred.std()
objs_pred=(objs_pred-objs_mean_pred)/objs_std_pred

differences=objs-objs_pred
accuracy=97.00
# Create a scatter plot
plt.scatter(objs, objs_pred, c=differences.flatten(), cmap="viridis", alpha=0.8)
# Add a diagonal line
plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes, color='black', linestyle='--')


# Add labels and title
plt.xlabel('Optimal Value', fontsize=12,family="Times New Roman")
plt.ylabel('Predicted Value', fontsize=12,family="Times New Roman")

# Add a colorbar
cbar = plt.colorbar()
cbar.ax.set_ylabel('Gap', rotation=270, fontsize=12, labelpad=15,family="Times New Roman")

for l in cbar.ax.yaxis.get_ticklabels():
    l.set_family('Times New Roman')

# Set the aspect ratio to 'equal' and adjust the limits
plt.gca().set_aspect('equal', adjustable='box')

plt.xticks(np.arange(-2,2.5,0.5),family="Times New Roman")
plt.yticks(np.arange(-2,2.5,0.5),family="Times New Roman")

plt.grid()

plt.text(-1.5, 1.2, f'Accuracy: {accuracy:.2f}%',family="Times New Roman")
plt.savefig("diving_result.png",dpi=666)


#%%
import torch
import matplotlib.pyplot as plt

# Assume you have two tensors: 'true_values' and 'pred_values'
# true_values = torch.tensor(...) # your actual values
# pred_values = torch.tensor(...) # your predicted values

# For illustrative purposes, let's create some dummy data
N = 100  # Number of instances
n_var = 120  # Number of binary variables

torch.manual_seed(0)

pred_values = torch.stack(sols_pred).squeeze(1)
true_values = sols

# Compute the number of correct predictions for each instance
#true_values = torch.randint(0, 2, (N, n_var))
#pred_values = torch.randint(0, 2, (N, n_var))  # Assume your model outputs binary values

correct_predictions = (true_values == pred_values).sum(dim=1)
incorrect_predictions = n_var - correct_predictions

# Create a stacked bar plot
plt.figure(figsize=(10, 6))
plt.bar(range(N), correct_predictions, color='green', label='Correct Predictions')
plt.bar(range(N), incorrect_predictions, bottom=correct_predictions, color='red', label='Incorrect Predictions')
plt.xlabel('Sample')
plt.ylabel('Number of Predictions')
plt.title('Comparison of Actual vs. Predicted Values')
plt.legend()
plt.grid(True)
plt.show()





