function [ r ] = ih_update( g,ih0,y,s,rho,m )
% update inverse Hessian for L-BFGS
% g -> gradient
% ih0 -> initial approximate inverse Hessian
% y,s,rho -> update history y_{k-m:k-1} and s_{k-m:k-1}
% r -> search direction r = ih*g
% m -> memory length

q = g;
a = zeros(m,1);

for i = m:-1:1
    a(i) = rho(i)*s(:,i)'*q;
    q = q - a(i)*y(:,i);
end
r = ih0*q;
for i = 1:1:m
    beta = rho(i)*y(:,i)'*r;
    r = r + s(:,i)*(a(i)-beta);
end

