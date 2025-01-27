function [D,J,GG] = F(m,Q,model)
% Forward operator
%
%   D = P^TA^{-1}(m)Q
%
% where P, Q encode the receiver and source locations and L is the first-order FD matrix
%
% use:
%   [D,J] = F(m,model);
%
% input:
%   m - squared-slownes [s^2/km^2]
%   Q - source weights, matrix of size ns x ns', where ns is the size of
%   the source grid (model.zs) and ns' determines the effective number of
%   sources.
%   model.h - gridspacing in each direction d = [d1, d2];
%   model.n - number of gridpoints in each direction n = [n1, n2]
%   model.f - frequency [Hz].
%   model.{zr,xr} - {z,x} locations of receivers [m] (must coincide with gridpoints)
%   model.{zs,xs} - {z,x} locations of sources [m] (must coincide with gridpoints)
%
%
% output:
%   D - data matrix
%   J - Jacobian as Spot operator

    % size
    nr = length(model.zr);
    ns = length(model.zs);
    nf = length(model.f);
    nx = prod(model.n);
    alpha = model.alpha;
    h = model.h / 1000;
    %  sampling operators
    Pr = getP(model.h,model.n,model.zr,model.xr);
    Ps = getP(model.h,model.n,model.zs,model.xs);
    
    % solve
    U = zeros(nx,ns,nf);
    D = zeros(nr,ns,nf);
    if model.data && nargout>2
        G = zeros(ns^2,ns^2,nf);
        L = getL(model.h,model.n); %first-order differentiation matrix
    end

    for k = 1:nf
        Ak = getA(model.f(k),m,model.h,model.n);
        U(:,:,k) = Ak\full(Ps'*Q);
        D(:,:,k) = model.w(k)*Pr*U(:,:,k);
        % % \dot{H}^1 inner product
        if model.data && nargout>2
            G(:,:,k) = kron(speye(ns),...
              inv(speye(ns)+h(1)*h(2)*conj(L*U(:,:,k))'*conj(L*U(:,:,k)/alpha)));
            GG = G; % single frequency;
            % GG = []; % if more frequencies are used.
            % for k = 1:nf
            %     GG = blkdiag(GG,G(:,:,k));
            % end
        end
        
    end
    


    D = D(:);
    
    % Jacobian
    J = opFunction(nr*ns*nf, nx, @(x,flag)Jmv(x,m,U,model,flag));
end


function y = Jmv(x,m,U,model,flag)
    % size
    nr = length(model.zr);
    ns = length(model.zs);
    nf = length(model.f);
    nx = prod(model.n);
    
    %% get matrices
    Pr = getP(model.h,model.n,model.zr,model.xr);
 
    %% compute mat-vec
    if flag == 1
        y = zeros(nr,ns,nf);
        for k = 1:nf
            Rk = zeros(nx,ns);
            Ak = getA(model.f(k),m,model.h,model.n);
            Gk = @(u)getG(model.f(k),m,u,model.h,model.n);
            for l = 1:ns
               Rk(:,l) = -Gk(U(:,l,k))*x;
            end
            y(:,:,k) = model.w(k)*Pr*(Ak\Rk);
        end
        y = y(:);
    else
        y = zeros(nx,1);
        x = reshape(x,[nr,ns,nf]);
        for k = 1:nf
            Ak = getA(model.f(k),m,model.h,model.n);
            Gk = @(u)getG(model.f(k),m,u,model.h,model.n);
            Rk = Ak'\(conj(model.w(k))*(Pr'*x(:,:,k)));
            for l = 1:size(U,2)
                y = y - Gk(U(:,l,k))'*Rk(:,l);
            end
        end
    end

end
