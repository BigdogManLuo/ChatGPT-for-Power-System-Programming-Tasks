% Define the ADMM function using YALMIP and Gurobi
function [x, z, y] = admm_solve_modified_yalmip_gurobi(A, B, c,x_var,z_var,Cons_x,Cons_z,f_x,g_z,rho, x0, z0, y0, max_iter, tol)
    
    % Initialize variables
    x = x0;
    z = z0;
    y = y0;

    % Iterate until convergence or maximum iterations reached
    for k = 1:max_iter
        
        w=sdpvar(2,1);
        t=sdpvar(1);
        
        % Update x
        x_prev = x;
        x_constraints=[Cons_x,A * x_var + B * z - c == w];
        x_constraints=[x_constraints,cone([t; (1/sqrt(rho))*w])];
        %x_objective = f_x + y'*(A * x_var + B * z - c) + 0.5 * rho * (A * x_var + B * z - c)' * (A * x_var + B * z - c);
        x_objective = f_x + y'*w + t;
        optimize(x_constraints, x_objective, sdpsettings('solver', 'gurobi'));
        x = value(x_var);

        w=sdpvar(2,1);
        t=sdpvar(1);
        
        % Update z
        z_prev = z;
        z_constraints=[Cons_z,A * x + B * z_var - c == w];
        z_constraints=[z_constraints,cone([t; (1/sqrt(rho))*w])];
        %z_objective = g_z + y'*(A * x + B * z_var - c) + 0.5 * rho * (A * x + B * z_var - c)' * (A * z_var + B * z - c);
        z_objective = g_z + y'*w+ t;
        optimize(z_constraints, z_objective, sdpsettings('solver', 'gurobi'));
        z = value(z_var);
        
        % Update y
        y = y + rho * (A * x + B * z - c);
        
       % Compute primal and dual residuals
        primal_residual = norm(A * x + B * z - c);
        %dual_residual = norm(rho * A' * (A * (x - x_prev) + B * (z - z_prev)));
        
        % Check for convergence
        if primal_residual < tol %&& dual_residual < tol
            fprintf("µÚ%.1f´ÎÑ­»·ÖÕÖ¹",k)
            break;
        end
        
    end
end

    