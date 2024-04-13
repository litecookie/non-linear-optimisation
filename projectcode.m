% ----------------------------------------------------------

% Project 2 - Financial maximisation problem
% Roll Number - 1010188967

% ----------------------------------------------------------

clear;

% ----------------------------------------------------------

% Predictor-corrector Primal Dual Interior Point method.
% max c'*x - (delta/2)*x'Sx
% s.t. A*x = b
% x >= 0

% Inputs: (n = 3)
% c = n*1 vector, objective coefficients
% A = 1*n matrix with m < n, A is full rank matrix
% b = 1*1 vector, RHS

% ----------------------------------------------------------
delta   = 4; % Between 3.5 and 4.5  

% Objective function coefficients
% Defined as c = -c to convert maximisation to minimisation problem
% Matrix S used in the matrix formulation for objective function

c = -[0.1073; 0.0737; 0.0627];  
S = -delta*[0.02778 0.00387 0.00021; 0.00387 0.01112 -0.00020; 0.00021 -0.00020 0.00115];  % Assume not used for a linear program, or check documentation/function comments

% Constraint function coefficients.

A = [1 1 1];
b = [1];

[m, n]  = size(A);
e       = ones(n,1);

%% Step 0: Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Obtain an initial interior solution [x(0), pie(0), z(0)]' 
% x(0)>0 and z(0)>0. Let k=0 and epsi be some small positive number
% (tolerance). Go to STEP 1.

k       = 0;        %counter
epsilon = 1/10^10;  %tolerance
eta     = .95;      %step length dampening constant
Aprime  = -eye(3);
bprime  = zeros(3,1);

% generate initial point from slide equations.

x_bar   = A'*((A*A')\b); 
pie_bar = (A*A')\(A*c);
z_bar   = c-A'*pie_bar;

del_x   = max([0; -1.5*min(x_bar)]);
del_z   = max([0; -1.5*min(z_bar)]);

del_x_bar   = del_x+.5*(x_bar+del_x*e)'*(z_bar+del_z*e)/sum(z_bar+del_z);
del_z_bar   = del_z+.5*(x_bar+del_x*e)'*(z_bar+del_z*e)/sum(x_bar+del_x);

x(:,k+1)    = x_bar+del_x_bar; %initial x(0), primal variable
pie(:,k+1)  = pie_bar; %initial pie(0), slack variable of dual
z(:,k+1)    = z_bar+del_z_bar; %initial z(0), dual variable

% Objective function values.
obj_pd(:,k+1)   = [c'*x(:,k+1)-1/2*x(:,k+1)'*S*x(:,k+1); 1/2*x(:,k+1)'*S*x(:,k+1) + b'*pie(:,k+1)];

% Norm of objective function and constraints to compute stop condition
Norm(:,k+1) = [norm(A*x(:,k+1)-b); norm(A'*pie(:,k+1)+z(:,k+1)-c); x(:,k+1)'*z(:,k+1);];
options     = optimset('Display', 'off');

while k>=0
 
 %% Step 1: Affine Scaling Direction Generation
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 % Solve KKT system for affine direction d_affine in the algorithm.

 %primal and dual residuals (updated based on objective function)
 r_p    = A*x(:,k+1)-b; 
 r_d    = A'*pie(:,k+1)+z(:,k+1)-c+S'*x(:,k+1); 

 X  = diag(x(:,k+1)); %diag(x(k))
 Z  = diag(z(:,k+1)); %diag(z(k))
 
 % Compute the KKT coefficient matrix
 kkt_coefficient=[S A' eye(size(A',1), size(X,2)); ...
 A zeros(m, size(A',2)) zeros(m, size(X,2)); ...
 Z zeros(size(X,1), size(A',2)) X];
 
 % Solve to compute affinity d variables
 d_aff      = -kkt_coefficient\[r_d; r_p; X*Z*e]; 
 d_x_affine = d_aff(1:n); %affine direction of x(k)
 d_z_affine = d_aff(n+m+1:end); %affine direction of z(k)
 
 %% Step 2: Centering Parameter Generation
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
 x_temp         = x(:,k+1);
 flag_x         = find(d_x_affine<0);
 alpha_x_affine = min([1; min(-x_temp(flag_x)./d_x_affine(flag_x))]);%alpha_x_affine
 z_temp         = z(:,k+1);
 flag_z         = find(d_z_affine<0);
 alpha_z_affine = min([1; min(-z_temp(flag_z)./d_z_affine(flag_z))]);%alpha_z_affine
 y(k+1)         = x(:,k+1)'*z(:,k+1)/n; 
 y_affine(k+1)  = ((x(:,k+1)+alpha_x_affine*d_x_affine)'*(z(:,k+1)+alpha_z_affine*d_z_affine))/n;%y_affine(k)
 tau(k+1)       = (y_affine(k+1)/y(k+1))^3;
 
 D_x    = diag(d_x_affine); 
 D_z    = diag(d_z_affine); 
 
 % Compute direction variable d using inverse matrix operation.
 d      = -kkt_coefficient\[r_d; r_p; X*Z*e-D_x*D_z*e-tau(k+1)*y(k+1)*e];%solve the KKT system
 d_x    = d(1:n); %d_x
 d_pie  = d(n+1:n+m); %d_pie
 d_z    = d(n+m+1:end); %d_z

 %% Step 3: New Primal and Dual solution Generation
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
 flag_x         = find(d_x<0);
 alpha_x_maximum= min([1; min(-x_temp(flag_x)./d_x(flag_x))]);% minimum ratio test for x
 alpha_x        = min([1; eta*alpha_x_maximum]); %alpha_x
 flag_z         = find(d_z<0);
 alpha_z_maximum= min([1; min(-z_temp(flag_z)./d_z(flag_z))]);%minimum ratio test for z
 alpha_z        = min([1; eta*alpha_z_maximum]); %alpha_z
 
 % update the counter
 k = k+1;
 
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 x(:,k+1)       = x(:,k)+alpha_x*d_x; %generate x(k+1)=x(k)+alpha_x*d_x
 pie(:,k+1)     = pie(:,k)+alpha_z*d_pie; %generate pie(k+1)=pie(k)+alpha_z*d_pie
 z(:,k+1)       = z(:,k)+alpha_z*d_z; %generate z(k+1)=z(k)+alpha_z*d_z
 obj_pd(:,k+1)  = [c'*x(:,k+1); b'*pie(:,k+1)];%primal and dual objective value
 Norm(:,k+1)    = [norm(A*x(:,k+1)-b); norm(A'*pie(:,k+1)+z(:,k+1)-c+S'*x(:,k+1));x(:,k+1)'*z(:,k+1);];
 
 % Condition for convergence.
 if isempty(find(Norm(:,k+1) >= epsilon))%if all norm of residual <= epsi, then optimal STOP.
    disp('Convergence Achieved')
    break
 end
end

%% Computed optimal solution and iterations displayed.

xsolution   = x(:,end); 
objval      = obj_pd(:,end); 

% Average of primal, dual objective values computed for optimal results
objvalf     = -(objval(1)+objval(2))/2;

% Display results.
disp('Optimal solution is found at:')
disp(xsolution')
disp('Maximum Expected Return:')
disp(objvalf)
T = table(x', pie');
T.Properties.VariableNames = {'Primal', 'Dual'};
disp(T);

%% Quadprog function in Matlab used to verify results.
 
[x2, obj_pd2] = quadprog(-S, c, Aprime, bprime, A, b, [],[],[], options);
xsolutionq=x2(:,end);

% Compute the optimal objective value
objval2 = c' * x2 - 0.5 * x2' * S * x2;
objval2 = -objval2;

% Display results
disp('Quadprog - Optimal solution is found at:')
disp(xsolutionq')
disp('Quadprog - Maximum expected return:')
disp(objval2)