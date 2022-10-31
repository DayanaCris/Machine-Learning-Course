function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

o = ones(m)(:,1);
pred= X*theta;
h = sigmoid(pred);
lg = log(o - h); 
J = 1/m * sum(-y' * diag(log(h)) - (o-y)' * diag(lg)) + lambda/(2*m)*sum(theta.^2);

%%%%%%%%%%%%%%%% GRadient %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

a = X(:,1);
b = (h - y)' * diag(a);
grad(1) =1/m * sum(b);  
for i=2:length(theta)
  x = X(:,i);
  v = (h - y)' * diag(x);
  grad(i) = 1/m * sum(v)+lambda/m*theta(i); 
endfor




% =============================================================

end
