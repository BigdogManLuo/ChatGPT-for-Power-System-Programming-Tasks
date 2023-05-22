clc
clear

x=sdpvar(2,1);
z=sdpvar(2,1);
y=sdpvar(2,1);

%决策变量x的约束
Cons_x=[];
Cons_x=[Cons_x,2*x(1)+3*x(2)==5]; %等式约束
Cons_x=[Cons_x,-1*x(1)+1*x(2)<=1];%不等式约束
Cons_x=[Cons_x,x>=0]; %下边界

%决策变量z的约束
Cons_z=[];
Cons_z=[Cons_z,2*z(1)-1*z(2)==0]; %等式约束
Cons_z=[Cons_z,0*z(1)-1*z(2)<=2];%不等式约束
Cons_z=[Cons_z,z>=0]; %下边界

%各自的目标函数
f_x=3*x(1)+2*x(2);
g_z=4*z(1)+5*z(2);

%连接点约束
A = [1, 2;3,4];
B = [2, 1;4,3];
c = [8;20];

%迭代初值
x0 = [0; 0];
z0 = [0; 0];
y0 = [0;0];

%ADMM参数
rho = 0.1;
max_iter = 30;
tol = 1e-3;

%求解
[x_optimal, z_optimal, y_optimal]=admm_solve_modified_yalmip_gurobi(A, B, c,x,z,Cons_x,Cons_z,f_x,g_z,rho, x0, z0, y0, max_iter, tol);

% Display the results
fprintf('Optimal x: [%.4f, %.4f]\n', x_optimal);
fprintf('Optimal z: [%.4f, %.4f]\n', z_optimal);
fprintf('Optimal y: [%.4f, %.4f]\n', y_optimal);

