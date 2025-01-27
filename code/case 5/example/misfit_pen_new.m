function [f,g,H,opt] = misfit_pen_new(m,D,lambda,model,alpha)
% Evaluate penalty misfit
%
% use:
%   [f,g,H] = phi_lambda(m,Q,D,lambda,model)
%

%%
if isfield(model,'mref')
    mref = model.mref;
else
    mref = 0*m;
end
if isfield(model,'mask')
    mask = model.mask;
else
    mask = 1;
end
%% get matrices
L = getL(model.h,model.n);
A = getA(model.f,m,model.h,model.n);
P = getP(model.h,model.n,model.zr,model.xr);
Q = getP(model.h,model.n,model.zs,model.xs);
G = @(u)getG(model.f,m,u,model.h,model.n);


ns = size(Q,1);
nr = size(P,1);
P = sparse(speye(nr)*P);
Q = sparse(speye(ns)*Q);
DD = reshape(D,nr,ns);
%% forward solve
%U = [sqrt(lambda)*A;P]\[sqrt(lambda)*P'*Q;D];
% B =  [sqrt(lambda)*A;P];
% b =  [sqrt(lambda)*Q';DD];
% U = B\b;
U = (lambda*(A'*A) + (P'*P))\(P'*DD + lambda*A'*Q');

%% adjoint field
V = lambda*(A*U - Q');

%% compute f
f = .5*norm(P*U - DD,'fro')^2 + .5*lambda*norm(A*U - Q'*speye(ns),'fro')^2 + .5*alpha*norm(L*m)^2;

%% compute g
if nargout>1
    if alpha
        g = alpha*(L'*L)*m;
    else 
        g=0;
    end

    for k = 1:ns
        g = g + real(G(U(:,k))'*V(:,k));
    end
    g = mask.*g;
end
%% get H
if nargout>2
    H = @(x)Hmv(x,m,U,alpha,lambda,model);
end
%% optimality
if nargout>3
    opt = [norm(g),  norm(A'*V - P'*(DD - P*U),'fro'), norm(A*U - Q','fro'), norm(m-mref), norm((DD - P*U),'fro')];
end

end

function y = Hmv(x,m,U,alpha,lambda,model)
%%
ns = size(U,2);
if isfield(model,'mask')
    mask = model.mask;
else
    mask = 1;
end
%% get matrices
L = getL(model.h,model.n);
A = getA(model.f,m,model.h,model.n);
P = getP(model.h,model.n,model.zr,model.xr);
G = @(u)getG(model.f,m,u,model.h,model.n);

%% compute mat-vec
y = mask.*x;
y = alpha*(L'*L)*y;

for k = 1:ns
    y = y + real(lambda*G(U(:,k))'*G(U(:,k))*x - lambda^2*G(U(:,k))'*A*((P*P' + lambda*(A'*A))\(A'*G(U(:,k))*x)));
end
y = mask.*y;
end
