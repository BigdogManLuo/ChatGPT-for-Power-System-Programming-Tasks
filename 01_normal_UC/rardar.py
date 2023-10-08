import numpy as np
import matplotlib.pyplot as plt
import matplotlib
 
matplotlib.rcParams['font.family']= ['Arial']    #字体设置
#matplotlib.rcParams['font.sans-serif'] = ['songNTR']
 
labels = np.array(["       model\n         robustness","code\n    robustness\n","model\nconsistency\n","code \nconsistency","code\n success rate","model\n success rate"])
dataLenth  = 6       # 数据长度

#原始数据
data_gpt35 = np.array([1,1, 2.33, 2.33, 2, 2]) #分别对应labels中的六个维度
data_gpt40 = np.array([3,3, 3, 3, 3, 3])
data_claude = np.array([0,0, 3, 3, 2, 2])
data_bard = np.array([0,0,3,3,0,0])

angles = np.linspace(0,2*np.pi,dataLenth,endpoint=False)   #根据数据长度平均分割圆周长

#闭合
data_gpt35 = np.concatenate((data_gpt35,[data_gpt35[0]]))
data_gpt40 = np.concatenate((data_gpt40,[data_gpt40[0]])) 
data_claude = np.concatenate((data_claude,[data_claude[0]])) 
data_bard = np.concatenate((data_bard,[data_bard[0]])) 

angles = np.concatenate((angles,[angles[0]]))
labels=np.concatenate((labels,[labels[0]]))  #对labels进行封闭



#画图
fig = plt.figure(facecolor="white",figsize=(12,8))       #facecolor 设置框体的颜色

plt.subplot(221,polar=True)
plt.title("ChatGPT3.5")
plt.plot(angles,data_gpt35,'bo-',color ='#3498db',linewidth=2)
plt.fill(angles,data_gpt35,facecolor='#3498db',alpha=0.25)    #填充两条线之间的色彩，alpha为透明度
plt.thetagrids(angles*180/np.pi,labels)          #做标签
plt.grid(True)
plt.yticks(np.arange(0,3.5,0.5))

plt.subplot(222,polar=True)
plt.title("ChatGPT4.0")
plt.plot(angles,data_gpt40,'bo-',color ='#e74c3c',linewidth=2)
plt.fill(angles,data_gpt40,facecolor='#e74c3c',alpha=0.25)    #填充两条线之间的色彩，alpha为透明度
plt.thetagrids(angles*180/np.pi,labels)          #做标签
plt.grid(True)
plt.yticks(np.arange(0,3.5,0.5))

plt.subplot(223,polar=True)
plt.title("Claude")
plt.plot(angles,data_claude,'bo-',color ='#2ecc71',linewidth=2)
plt.fill(angles,data_claude,facecolor='#2ecc71',alpha=0.25)    #填充两条线之间的色彩，alpha为透明度
plt.thetagrids(angles*180/np.pi,labels)          #做标签
plt.grid(True)
plt.yticks(np.arange(0,3.5,0.5))

plt.subplot(224,polar=True)
plt.title("Bard")
plt.plot(angles,data_bard,'bo-',color ='#f1c40f',linewidth=2)
plt.fill(angles,data_bard,facecolor='#f1c40f',alpha=0.25)    #填充两条线之间的色彩，alpha为透明度
plt.thetagrids(angles*180/np.pi,labels)          #做标签
plt.grid(True)
plt.yticks(np.arange(0,3.5,0.5))
fig.subplots_adjust(hspace=0.35,wspace=0.2)

plt.savefig("rardar.png",dpi=1024)
plt.show()