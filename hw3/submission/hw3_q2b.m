% Define symbolic variables
syms x1 x2

% Define the function f
f = 100*x1^4 + 0.01*x2^4;

% Calculate gradient of f
grad_f = gradient(f, [x1, x2]);

% Initial guess
x_k = [1; 1]; % Column vector [x1; x2]

% Constants for backtracking
beta = 0.5;
gamma = 0.5;
alpha_k = 1;

% Maximum iterations
max_iterations = 10000;
iteration_counter = 1;

% Perform Newton's backtracking algorithm
while iteration_counter <= max_iterations
    % Calculate gradient at current x_k
    grad_val = vpa(subs(grad_f, [x1, x2], x_k.'), 200);

    % Calculate d_k
    d_k = vpa(-grad_val, 200);
    
    if norm(d_k) <= 10e-3
        break;
    end
    
    % Calculate f(x_k)
    f_xk = vpa(subs(f, [x1, x2], x_k.'), 200);
    
    % Compute alpha_k using backtracking line search
    while true
        % Compute x_k_plus_1 and f(x_k_plus_1)
        x_k_plus_1 = x_k + alpha_k*d_k;
        f_xk_plus_1 = vpa(subs(f, [x1, x2], x_k_plus_1.'), 200);
        
        % Compute RHS of backtracking condition
        rhs_backtracking = vpa(gamma*alpha_k*grad_val.'*d_k, 200);
        
        % Check backtracking condition
        if f_xk - f_xk_plus_1 >= -rhs_backtracking
            break; % Exit backtracking loop
        else
            alpha_k = beta*alpha_k; % Update alpha_k
        end
    end
    
    % Update x_k for the next iteration
    x_k = vpa(x_k_plus_1, 200);
    
    % Increment iteration counter
    iteration_counter = iteration_counter + 1;
    
    disp(iteration_counter);
    disp(vpa(subs(f, [x1, x2], x_k.'), 30))
end

disp(['Number of iterations for convergence: ', num2str(iteration_counter)]);
disp(['Final solution: x = [', char(x_k(1)), ', ', char(x_k(2)), ']']);
