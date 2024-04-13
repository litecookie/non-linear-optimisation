syms x1 x2 x3

f1 = x1^2 + x2^2 + x3^2;
newtonsMethodBacktracking(f1, [1; 1; 1], 0.5, 0.5, 1000);

f2 = x1^2 + 2*x2^2 - 2*x1*x2 - 2*x2;
newtonsMethodBacktracking(f2, [0; 0], 0.5, 0.5, 1000);

f3 = 100*(x2 - x1^2)^2 + (1 - x1)^2;
newtonsMethodBacktracking(f3, [-1.2; 1], 0.5, 0.5, 1000);

f4 = (x1+x2)^4 + x2^2;
newtonsMethodBacktracking(f4, [2; -2], 0.5, 0.5, 1000);

f5_1 = (x1 - 1)^2 + (x2 - 1)^2 + (x1^2 + x2^2 - 0.25)^2;
newtonsMethodBacktracking(f5_1, [1; -1], 0.5, 0.5, 1000);

f5_2 = (x1 - 1)^2 + (x2 - 1)^2 + 10*(x1^2 + x2^2 - 0.25)^2;
newtonsMethodBacktracking(f5_2, [1; -1], 0.5, 0.5, 1000);

f5_3 = (x1 - 1)^2 + (x2 - 1)^2 + 100*(x1^2 + x2^2 - 0.25)^2;
newtonsMethodBacktracking(f5_3, [1; -1], 0.5, 0.5, 1000);

function newtonsMethodBacktracking(f, x0, beta, gamma, maxIterations)
    syms x1 x2 x3

    if length(x0) == 3
        x = [x1 x2 x3];
    elseif length(x0) == 2
        x = [x1 x2];
    end

    % Calculate gradient of f
    grad_f = gradient(f, x);
    
    % Constants for backtracking
    alpha_k = 1;
    
    % Initialize
    x_k = x0;
    iteration_counter = 1;
    
    % Perform Newton's backtracking algorithm
    while iteration_counter <= maxIterations
        % Calculate gradient at current x_k
        grad_val = vpa(subs(grad_f, x, x_k.'), 200);
        fx_val = vpa(subs(f, x, x_k.'), 200);
        
        if norm(grad_val)/(1+norm(fx_val)) <= 10e-5
            break;
        end

        d_k = vpa(-grad_val, 200);  % Assuming Hessian is identity for gradient descent
        f_xk = vpa(subs(f, x, x_k.'), 200);

        while true
            x_k_plus_1 = x_k + alpha_k * d_k;
            f_xk_plus_1 = vpa(subs(f, x, x_k_plus_1.'), 200);
            rhs_backtracking = vpa(gamma * alpha_k * (grad_val.') * d_k, 200);

            if f_xk - f_xk_plus_1 >= -rhs_backtracking
                break;
            else
                alpha_k = beta * alpha_k;
            end
        end

        x_k = vpa(x_k_plus_1, 200);
        iteration_counter = iteration_counter + 1;
        % Store current iteration result
        iterations(:, iteration_counter) = [x_k; d_k; alpha_k];
    end

    disp(['Number of iterations for convergence: ', num2str(iteration_counter)]);
    disp(['Final solution: x = [', char(x_k.'), ']']);

    
    % Choose display format based on iteration count
    if iteration_counter > 15
        % Prepare table for the first 10 iterations
        Iteration_fcount = (1:10)';
        X1_First = round(iterations(1, 1:10)', 6);
        X2_First = round(iterations(2, 1:10)', 6);
        d_k_First = round(iterations(3:4, 1:10)', 6);
        alpha_k_First = round(iterations(5, 1:10)', 6);
        T_First = table(Iteration_fcount, X1_First, X2_First, d_k_First, alpha_k_First);
        
        % Prepare table for the last 5 iterations
        Iteration_lcount = (iteration_counter-4:iteration_counter)';
        X1_Last = round(iterations(1, end-4:end)', 6);
        X2_Last = round(iterations(2, end-4:end)', 6);
        d_k_Last = round(iterations(3:4, end-4:end)', 6);
        alpha_k_Last = round(iterations(5, end-4:end)', 6);
        T_Last = table(Iteration_lcount, X1_Last, X2_Last, d_k_Last, alpha_k_Last);
        
        disp('First 10 iterations:');
        disp(T_First);
        disp('Last 5 iterations:');
        disp(T_Last);
    else
        % Prepare table for all iterations
        IterationsAll = (1:iteration_counter)';
        X1All = round(iterations(1, :)', 6);
        X2All = round(iterations(2, :)', 6);
        d_k_All = round(iterations(3:4, :)', 6);
        alpha_k_All = round(iterations(5, :)', 6);
        TAll = table(IterationsAll, X1All, X2All, d_k_All, alpha_k_All);
        
        disp('All iterations:');
        disp(TAll);
    end
end
