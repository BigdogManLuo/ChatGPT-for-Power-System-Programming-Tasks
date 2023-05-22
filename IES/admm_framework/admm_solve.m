
% Define the modified admm_solve function with bounds, linear inequality constraints, and additional equality constraints for x and z
function [x, z, y] = admm_solve(f, g, A, B, c, rho, x0, z0, y0, lb_x, lb_z, D, e, F, h, A_eq, b_eq, C_eq, d_eq, max_iter, tol)
    % Initialize variables
    x = x0;
    z = z0;
    y = y0;

    % Iterate until convergence or maximum iterations reached
    for k = 1:max_iter
        % Update x
        x_prev = x;
        x = fmincon(@(x) augmented_lagrangian(x, z, y, A, B, c, rho, f, g), x, D, e, A_eq, b_eq, lb_x, [], [], []);

        % Update z
        z_prev = z;
        z = fmincon(@(z) augmented_lagrangian(x, z, y, A, B, c, rho, f, g), z, F, h, C_eq, d_eq, lb_z, [], [], []);

        % Update y
        y = y + rho * (A * x + B * z - c);
        
        
        % Compute primal and dual residuals
        primal_residual = norm(A * x + B * z - c);
        dual_residual = norm(rho * A' * (A * (x - x_prev) + B * (z - z_prev)));
        
        % Check for convergence
        if primal_residual < tol && dual_residual < tol
            fprintf("µÚ%.2f´ÎÑ­»·ÖÕÖ¹",k)
            break;
        end

    end
end















