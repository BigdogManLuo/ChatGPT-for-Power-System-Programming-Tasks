function L = augmented_lagrangian(x, z, y, A, B, c, rho, f, g)
    L = f(x) + g(z) + y' * (A * x + B * z - c) + (rho / 2) * norm(A * x + B * z - c)^2;
end
