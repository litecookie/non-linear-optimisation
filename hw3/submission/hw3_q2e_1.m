% Problem set 3 - Question 2 - Part c

% Define symbolic variables
syms x1 x2

% Define the function f
f = sqrt(x1^2+1) + sqrt(x2^2+1);

% Calculate gradient of f
grad_f = gradient(f, [x1, x2]);

% Calculate Hessian of f
hessian_f = hessian(f, [x1, x2]);

% Initial guess (you may change this based on your problem)
x_k = [10; 10]; % Column vector [x1; x2]

% Display settings
format long

% Maximum iterations
max_iterations = 15;
iteration_counter = 1;

% Store iteration results
iterations = [];

pd_counter = 0; % Counter for positive definite Hessian occurrences

% Calculate first iteration values, to start Newton's method
grad_val = vpa(subs(grad_f, [x1, x2], x_k.'), 500);
hessian_val = vpa(subs(hessian_f, [x1, x2], x_k.'), 500);
d_k = vpa(-inv(hessian_val)*grad_val, 500); % Equivalent to inv(H)*grad
x_k = x_k + d_k;
iterations(:, iteration_counter) = x_k;

% Check if the initial Hessian is positive definite
if all(eig(hessian_val) > 0)
    pd_counter = pd_counter + 1; % Increment if positive definite
end

% Perform check and continue Newton's iterative method.
while true
    % Break condition (You might need a proper convergence check)
    if norm(grad_val) <= 10e-6
        break;
    end
    
    % Check for singularity of Hessian matrix
    if cond(hessian_val) > 1e10
        disp('Hessian matrix is nearly singular. Terminating iteration.');
        break;
    end

    iteration_counter = iteration_counter + 1;
    
    % Calculate gradient and Hessian at current x_k
    grad_val = vpa(subs(grad_f, [x1, x2], x_k.'), 500);
    hessian_val = vpa(subs(hessian_f, [x1, x2], x_k.'), 500);
    
    % Check if the initial Hessian is positive definite
    if all(eig(hessian_val) > 0)
        pd_counter = pd_counter + 1; % Increment if positive definite
    end

    % Calculate d_k
    d_k = vpa(-inv(hessian_val)*grad_val, 500); % Equivalent to inv(H)*grad
    
    % Update x_k
    x_k = x_k + d_k;
    
    % Store current iteration result
    iterations(:, iteration_counter) = x_k;
    disp(iteration_counter);
    disp(vpa(subs(f, [x1, x2], x_k.'), 30))
end

disp('Number of iterations completed:');
disp(iteration_counter);