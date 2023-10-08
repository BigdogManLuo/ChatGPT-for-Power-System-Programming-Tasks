import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family']='Arial'     
matplotlib.rcParams['font.sans-serif'] = ['songNTR']

x=[1,2,3,4]

#原始数据

#----------------------gpt3.5-----------------------#
y_gpt35_p1_modelcor=[0,0,0,1/3] #Model correctness   四个数据分别对应四次迭代
y_gpt35_p1_codecor=[0,0,0,1/3]  #Code correctness

y_gpt35_p2_modelcor=[0,1/3,1/3,2/3] #Model correctness
y_gpt35_p2_codecor=[1/3,1/3,1/3,2/3]  #Code correctnes

y_gpt35_p3_modelcor=[1/3,2/3,1,1] #Model correctness
y_gpt35_p3_codecor=[1/3,1/3,1,1]  #Code correctnes

#----------------------gpt4.0-----------------------#
y_gpt40_p1_modelcor=[0,1,1,1] #Model correctness
y_gpt40_p1_codecor=[1,1,1,1]  #Code correctness

y_gpt40_p2_modelcor=[0,1,1,1] #Model correctness
y_gpt40_p2_codecor=[1,1,1,1]  #Code correctnes

y_gpt40_p3_modelcor=[2/3,1,1,1] #Model correctness
y_gpt40_p3_codecor=[1,1,1,1]  #Code correctnes

#----------------------claude-----------------------#
y_claude_p1_modelcor=[0,0,0,0] #Model correctness
y_claude_p1_codecor=[0,0,0,0]  #Code correctness

y_claude_p2_modelcor=[0,0,1/3,1] #Model correctness
y_claude_p2_codecor=[1,1,1,1]  #Code correctnes

y_claude_p3_modelcor=[1,1,1,1] #Model correctness
y_claude_p3_codecor=[1,1,1,1]  #Code correctnes

#----------------------google bard-----------------------#
y_bard_p1_modelcor=[0,0,0,0] #Model correctness
y_bard_p1_codecor=[0,0,0,0]  #Code correctness

y_bard_p2_modelcor=[0,0,0,0] #Model correctness
y_bard_p2_codecor=[0,0,0,0]  #Code correctnes

y_bard_p3_modelcor=[0,0,0,0] #Model correctness
y_bard_p3_codecor=[0,0,0,0]  #Code correctnes


#--------------------------Figure---------------------#
fig = plt.figure(facecolor="white",figsize=(11,5.5))  
plt.subplot(2,3,1)
plt.plot(x,y_gpt35_p1_modelcor,label="ChatGPT3.5",color="#0c2461",linestyle="--",marker="o",markersize=4)
plt.plot(x,y_gpt40_p1_modelcor,label="ChatGPT4.0",color="#B33771",marker="*",markersize=4)
plt.plot(x,y_claude_p1_modelcor,label="Claude",color="#079992",linestyle="-.",marker="^",markersize=4)
plt.plot(x,y_bard_p1_modelcor,label="Bard",color="#60a3bc",linestyle=":",marker="x",markersize=4)
plt.xlabel("Iteration times")
plt.ylabel("Correctness")
plt.title("Simple description")
plt.xticks(np.arange(1,5),labels=[0,1,2,3])
plt.grid()


plt.subplot(2,3,2)
plt.plot(x,y_gpt35_p2_modelcor,label="ChatGPT3.5",color="#0c2461",linestyle="--",marker="o",markersize=4)
plt.plot(x,y_gpt40_p2_modelcor,label="ChatGPT4.0",color="#B33771",marker="*",markersize=4)
plt.plot(x,y_claude_p2_modelcor,label="Claude",color="#079992",linestyle="-.",marker="^",markersize=4)
plt.plot(x,y_bard_p2_modelcor,label="Bard",color="#60a3bc",linestyle=":",marker="x",markersize=4)
plt.xlabel("Iteration times")
plt.ylabel("Correctness")
plt.title("Intermediate description")
plt.xticks(np.arange(1,5),labels=[0,1,2,3])
plt.grid()

plt.subplot(2,3,3)
plt.plot(x,y_gpt35_p3_modelcor,label="ChatGPT3.5",color="#0c2461",linestyle="--",marker="o",markersize=4)
plt.plot(x,y_gpt40_p3_modelcor,label="ChatGPT4.0",color="#B33771",marker="*",markersize=4)
plt.plot(x,y_claude_p3_modelcor,label="Claude",color="#079992",linestyle="-.",marker="^",markersize=4)
plt.plot(x,y_bard_p3_modelcor,label="Bard",color="#60a3bc",linestyle=":",marker="x",markersize=4)
plt.xlabel("Iteration times")
plt.ylabel("Correctness")
plt.title("Sophisticated description")
plt.xticks(np.arange(1,5),labels=[0,1,2,3])
plt.grid()

plt.subplot(2,3,4)
plt.plot(x,y_gpt35_p1_codecor,label="ChatGPT3.5",color="#0c2461",linestyle="--",marker="o",markersize=4)
plt.plot(x,y_gpt40_p1_codecor,label="ChatGPT4.0",color="#B33771",marker="*",markersize=4)
plt.plot(x,y_claude_p1_codecor,label="Claude",color="#079992",linestyle="-.",marker="^",markersize=4)
plt.plot(x,y_bard_p1_codecor,label="Bard",color="#60a3bc",linestyle=":",marker="x",markersize=4)
plt.xlabel("Iteration times")
plt.ylabel("Correctness")
plt.xticks(np.arange(1,5),labels=[0,1,2,3])
plt.grid()

plt.subplot(2,3,5)
plt.plot(x,y_gpt35_p2_codecor,label="ChatGPT3.5",color="#0c2461",linestyle="--",marker="o",markersize=4)
plt.plot(x,y_gpt40_p2_codecor,label="ChatGPT4.0",color="#B33771",marker="*",markersize=4)
plt.plot(x,y_claude_p2_codecor,label="Claude",color="#079992",linestyle="-.",marker="^",markersize=4)
plt.plot(x,y_bard_p2_codecor,label="Bard",color="#60a3bc",linestyle=":",marker="x",markersize=4)
plt.xlabel("Iteration times")
plt.ylabel("Correctness")
plt.xticks(np.arange(1,5),labels=[0,1,2,3])
plt.grid()

plt.subplot(2,3,6)
plt.plot(x,y_gpt35_p3_codecor,label="ChatGPT3.5",color="#0c2461",linestyle="--",marker="o",markersize=4)
plt.plot(x,y_gpt40_p3_codecor,label="ChatGPT4.0",color="#B33771",marker="*",markersize=4)
plt.plot(x,y_claude_p3_codecor,label="Claude",color="#079992",linestyle="-.",marker="^",markersize=4)
plt.plot(x,y_bard_p3_codecor,label="Bard",color="#60a3bc",linestyle=":",marker="x",markersize=4)
plt.xlabel("Iteration times")
plt.ylabel("Correctness")
plt.xticks(np.arange(1,5),labels=[0,1,2,3])
plt.grid()
plt.legend(ncol=4,bbox_to_anchor=(0.225,2.75))
plt.text(4.5,2,"   Model\nCorrectness")
plt.text(4.5,0.35,"   Code\nCorrectness")
fig.subplots_adjust(wspace=0.4,hspace=0.4)
plt.savefig("iteration.png",dpi=1024)
plt.show()

