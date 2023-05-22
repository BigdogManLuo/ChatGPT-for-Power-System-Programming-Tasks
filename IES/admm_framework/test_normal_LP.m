clc
clear

%������߱���
x=sdpvar(2,1);
z=sdpvar(2,1);

%�������
f=3*x(1)+2*x(2);
g=4*z(1)+5*z(2);

%����Ŀ�꺯��
obj=f+g;

%����Լ��
Cons=[];
Cons=[Cons,-1*x(1)+x(2)<=1];
Cons=[Cons,0*z(1)-z(2)<=2];
Cons=[Cons,2*x(1)+3*x(2)==5];
Cons=[Cons,2*z(1)-1*z(2)==0];

Cons=[Cons,1*x(1)+2*x(2)+2*z(1)+1*z(2)==8];
Cons=[Cons,3*x(1)+4*x(2)+4*z(1)+3*z(2)==20];
Cons=[Cons,x>=0];
Cons=[Cons,z>=0];
%���
options = sdpsettings('solver', 'gurobi','showprogress',1);
optimize(Cons, obj, options);

%��ȡ���߱���ȡֵ
x_optimal = value(x);
z_optimal = value(z);

%��ȡ����Ŀ�꺯��ֵ
optimalObj=value(obj);

