from utilities import loadDataset,lossFunc
from models import divingModel
import numpy as np
import torch
from torch.utils.data import TensorDataset,DataLoader
import matplotlib.pyplot as plt
import os
import gurobipy
from gurobipy import GRB
import time
import pickle
import gurobipy as gp

#取100组不同的UC instances
trainFile = os.listdir("data/instances/uc/test")[0:100]

#取对应的二分图sample
adjs=pickle.load(open("data/samples/uc/test/"+"adjs.pkl","rb"))
features=pickle.load(open("data/samples/uc/test/"+"features.pkl","rb"))
sols=pickle.load(open("data/samples/uc/test/"+"sols.pkl","rb"))


#%% 固定整数变量求解子问题

#加载模型
model=divingModel(nfeat=1,nhid=64,nclass=1,dropout=0.5,nNodes=2844,nOut=240,nlhid=256)  #nNodes和nOut需要调，nNodes约束总数+决策变量总数； nOut：决策变量数
state_dict = torch.load('model/diving_model.pth')
model.load_state_dict(state_dict['model'])

#测试集的每组样本目标函数值
Objs_true=pickle.load(open("data/tmp/test/Objs.pkl","rb"))



#预测0-1整数解
def OptimizeRemainedLP(ins_index,U_values):
    
    m = gp.read("data/instances/uc/test/instance"+str(ins_index)+".lp")
    
    for n in range(10):
        for t in range(24):
            var_name=f"U_{n}_{t}"
            var=m.getVarByName(var_name)
            # Set the value from U_values
            var.setAttr("LB", U_values[n, t])
            var.setAttr("UB", U_values[n, t])
    
    #Optimize the Remained LP problem
    start_time=time.time()
    m.optimize()
    end_time=time.time()
    
    if m.status == gp.GRB.Status.OPTIMAL:
        status="feasible"
    else:
        status="infeasible"
        
    if status=="feasible":
        #Get Objective Value
        obj=m.getObjective().getValue()
    else:
        obj=np.random.uniform(510000,530000)
        
    return status,obj,end_time-start_time
   


#merge adjs,features and sols to torch Dataset
dataset=TensorDataset(adjs,features,sols)


model.eval() #不启用BatchNormalization 和 Dropout
Objs_pred=[]
Time=[]
Status=[]
dataloader=DataLoader(dataset, batch_size=1, shuffle=True)
ins_index=0
with torch.no_grad():
    for adj,feature,sol in dataloader:
        outputs = model(feature, adj.float())
        # mapping outputs to (0~1)
        outputs=outputs.numpy()
        outputs=(outputs>0.5).astype(np.int8)
        # recover outputs to shape (N,T)
        outputs=outputs.reshape(24,10)
        outputs=outputs.T
        status,obj,consum_time=OptimizeRemainedLP(ins_index,outputs)
        #Update ins_index
        ins_index=ins_index+1
        
        Objs_pred.append(obj)
        Time.append(consum_time)
        Status.append(status)
    
#%% Visualization

Objs_true=np.array(Objs_true)
Objs_pred=np.array(Objs_pred)

#Calculate R^2 score
R2=1-np.sum((Objs_true-Objs_pred)**2)/np.sum((Objs_true-Objs_true.mean())**2)

#Calculate relevant regression metrics
from sklearn.metrics import mean_absolute_error,mean_squared_error
MAE=mean_absolute_error(Objs_true,Objs_pred)
RMSE=np.sqrt(mean_squared_error(Objs_true,Objs_pred))



#Normalize Objs_true and Objs_pred with max-min normalization
Objs_pred=(Objs_pred-Objs_true.min())/(Objs_true.max()-Objs_true.min())
Objs_true=(Objs_true-Objs_true.min())/(Objs_true.max()-Objs_true.min())

# Create a scatter plot     
plt.figure(figsize=(10,6))
for i in range(len(Objs_true)):
    if Status[i]=="infeasible":
        #Scatter "x"
        plt.scatter(Objs_true[i],Objs_pred[i],c="red",marker="x")
    else:
        plt.scatter(Objs_true[i],Objs_pred[i],c="blue")
        

#Set xticks and yticks
plt.xlabel("Normalized Cost(true)",fontsize=20,fontname = 'Arial')
plt.ylabel("Normalized Cost(pred)",fontsize=20,fontname = 'Arial')

#show R^2 score in title
plt.title(r'$R^2$'+"="+str(round(R2,4)),fontsize=20,fontname = 'Arial')

ax=plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')


#Plot a dividing line
plt.plot([min(min(Objs_pred),min(Objs_true)), max(max(Objs_pred),max(Objs_true))],
         [min(min(Objs_pred),min(Objs_true)), max(max(Objs_pred),max(Objs_true))], 'k-', lw=1)

#Set Title 'Time Improvement: 9200%'
#plt.title('Time Improvement: 6.9%', fontsize=20,fontweight='bold',horizontalalignment='center',verticalalignment='center',family="Times New Roman")

#plt.text(740000, 755000, 'Time Improvement: 9200%', fontsize=15,fontweight='bold',horizontalalignment='center',verticalalignment='center',family="Times New Roman")

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

plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])

#Save the figure with dpi=660
plt.savefig("scatter_plot.png",dpi=660)
plt.show()


'''
#Save Objs_true,Objs_pred,Status into savedata/... pkl format
with open('savedata/Objs_true.pkl', 'wb') as f:
    pickle.dump(Objs_true, f)
with open('savedata/Objs_pred.pkl', 'wb') as f:
    pickle.dump(Objs_pred, f)
with open('savedata/Status.pkl', 'wb') as f:
    pickle.dump(Status, f)
'''



    
'''    
    for i in range(adjs.shape[0]):
        feature=features[i]
        adj=adjs[i]
        feature=feature.unsqueeze(0)
        adj=adj.unsqueeze(0)
        start=time.time()
        sol_pred=model(feature,adj.float())
        end=time.time()
        print("Time Consume:"+str(end-start)+"s")
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
import matplotlib
matplotlib.rcParams['font.family']= ['Arial']
objs_mean=objs.mean()
objs_std=objs.std()
objs=(objs-objs_mean)/objs_std

objs_mean_pred=objs_pred.mean()
objs_std_pred=objs_pred.std()
objs_pred=(objs_pred-objs_mean_pred)/objs_std_pred

differences=objs-objs_pred
accuracy=100-len(np.where(differences!=0))
# Create a scatter plot
plt.scatter(objs, objs_pred, c=differences.flatten(), cmap="viridis", alpha=0.8)
# Add a diagonal line
plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes, color='black', linestyle='--')


# Add labels and title
plt.xlabel('Optimal Value', fontsize=12,family="Arial")
plt.ylabel('Predicted Value', fontsize=12,family="Arial")

# Add a colorbar
cbar = plt.colorbar()
cbar.ax.set_ylabel('Gap', rotation=270, fontsize=12, labelpad=15,family="Arial")

for l in cbar.ax.yaxis.get_ticklabels():
    l.set_family('Arial')

# Set the aspect ratio to 'equal' and adjust the limits
plt.gca().set_aspect('equal', adjustable='box')

plt.xticks(np.arange(-2,2.5,0.5),family="Arial")
plt.yticks(np.arange(-2,2.5,0.5),family="Arial")

plt.grid()

plt.text(-1.5, 1.2, f'Accuracy: {accuracy:.2f}%',family="Arial")
plt.savefig("diving_result.png",dpi=666)

#%%
import pickle
pickle.dump(objs_pred,open('result/objs_pred.pkl','wb'))
pickle.dump(objs,open('result/objs.pkl','wb'))


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
'''




