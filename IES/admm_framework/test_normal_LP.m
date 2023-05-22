clc
clear

%定义决策变量
x=sdpvar(2,1);
z=sdpvar(2,1);

%定义参数
f=3*x(1)+2*x(2);
g=4*z(1)+5*z(2);

%定义目标函数
obj=f+g;

%定义约束
Cons=[];
Cons=[Cons,-1*x(1)+x(2)<=1];
Cons=[Cons,0*z(1)-z(2)<=2];
Cons=[Cons,2*x(1)+3*x(2)==5];
Cons=[Cons,2*z(1)-1*z(2)==0];

Cons=[Cons,1*x(1)+2*x(2)+2*z(1)+1*z(2)==8];
Cons=[Cons,3*x(1)+4*x(2)+4*z(1)+3*z(2)==20];
Cons=[Cons,x>=0];
Cons=[Cons,z>=0];
%求解
options = sdpsettings('solver', 'gurobi','showprogress',1);
optimize(Cons, obj, options);

%获取决策变量取值
x_optimal = value(x);
z_optimal = value(z);

%获取最优目标函数值
optimalObj=value(obj);

