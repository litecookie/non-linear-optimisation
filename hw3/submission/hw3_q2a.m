% Problem set 3 - Question 2 - Part a

% Define symbolic variables
syms x1 x2

% Define the function f
f = 100*x1^4 + 0.01*x2^4;

% Calculate gradient of f
grad_f = gradient(f, [x1, x2]);

% Calculate Hessian of f
hessian_f = hessian(f, [x1, x2]);

% Initial guess (you may change this based on your problem)
x_k = [1; 1]; % Column vector [x1; x2]

% Display settings
format long

% Maximum iterations
max_iterations = 15;
iteration_counter = 1;

% Store iteration results
iterations = [];

pd_counter = 0; % Counter for positive definite Hessian occurrences

% Calculate first iteration values, to start Newton's method
grad_val = double(subs(grad_f, [x1, x2], x_k.'));
hessian_val = double(subs(hessian_f, [x1, x2], x_k.'));
d_k = -inv(hessian_val)*grad_val; % Equivalent to inv(H)*grad
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

    iteration_counter = iteration_counter + 1;
    
    % Calculate gradient and Hessian at current x_k
    grad_val = double(subs(grad_f, [x1, x2], x_k.'));
    hessian_val = double(subs(hessian_f, [x1, x2], x_k.'));
    
    % Check if the initial Hessian is positive definite
    if all(eig(hessian_val) > 0)
        pd_counter = pd_counter + 1; % Increment if positive definite
    end

    % Calculate d_k
    d_k = -inv(hessian_val)*grad_val; % Equivalent to inv(H)*grad
    
    % Update x_k
    x_k = x_k + d_k;
    
    % Store current iteration result
    iterations(:, iteration_counter) = x_k;
end

% Choose display format based on iteration count
if iteration_counter > 15
    % Prepare table for the first 10 iterations
    Iteration_fcount = (1:10)';
    X1_First = iterations(1, 1:10)';
    X2_First = iterations(2, 1:10)';
    T_First = table(Iteration_fcount, X1_First, X2_First);
    
    % Prepare table for the last 5 iterations
    Iteration_lcount = (iteration_counter-4:iteration_counter)';
    X1_Last = iterations(1, end-4:end)';
    X2_Last = iterations(2, end-4:end)';
    T_Last = table(Iteration_lcount, X1_Last, X2_Last);
    
    disp('First 10 iterations:');
    disp(T_First);
    disp('Last 5 iterations:');
    disp(T_Last);
else
    % Prepare table for all iterations
    IterationsAll = (1:iteration_counter)';
    X1All = iterations(1, :)';
    X2All = iterations(2, :)';
    TAll = table(IterationsAll, X1All, X2All);
    
    disp('All iterations:');
    disp(TAll);
end

disp('Number of iterations for convergence:');
disp(iteration_counter);

if pd_counter == iteration_counter
    disp('Hessian matrices are positive definite in each iteration until convergence');
else 
    disp('Hessian matrices were not always positive definite');
end    