function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
w=size(g);
if length(g)>1
    l=prod(w(:));
else l=length(g);
end

for i=1:l
    g(i)=1/(1+exp(1)^(-z(i)));
end

% =============================================================

end
