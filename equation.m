% Define the matrices T, L, and d
% Example matrices (you can replace these with your specific matrices)
T = [2.001, 3, 4; 3, 5.001, 7; 4, 7, 10.001];
L = [1, 0, 0; 0, 1, 0; 0, 0, 1];
d = [20.0019; 34.0004; 48.0202];

% Perform the matrix operation
% (T' * T + L' * L)⁻¹ * d' * T

result = inv(T' * T + L' * L) * (d' * T)';
result2 = inv(T' * T) * (d' * T)';

% Display the result
disp(result);
disp(result2);
