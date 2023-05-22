clc
clear

% Define matrices A_eq, C_eq, and vectors b_eq, d_eq for the additional equality constraints
A_eq = [2, 3];
b_eq = [5];

C_eq = [2, -1];
d_eq = [0];

% Define matrices D, F, and vectors e, g for the linear inequality constraints
D = [-1, 1];
e = [1];

F = [0, -1];
h = [2];

% Define functions f(x) and g(z)
c1 = [3; 2];
c2 = [4; 5];
f = @(x) c1' * x;
g = @(z) c2' * z;

% Define matrices A, B, and vector c
A = [1, 2];
B = [2, 1];
c = [8];

% Set initial guesses for x, z, and y
x0 = [0; 0];
z0 = [1; 2];
y0 = 0;

% Set ADMM parameters
rho = 0.01;
max_iter = 100;
tol = 1e-4;

% Set the lower bounds for x and z
lb_x = [0; 0];
lb_z = [0; 0];

% Solve the problem using ADMM
[x, z, y] = admm_solve(f, g, A, B, c, rho, x0, z0, y0, lb_x, lb_z, D, e, F, h, A_eq, b_eq, C_eq, d_eq, max_iter, tol);

% Display the results
fprintf('Optimal x: [%.4f, %.4f]\n', x);
fprintf('Optimal z: [%.4f, %.4f]\n', z);
fprintf('Optimal y: %.4f\n', y);


