function [f,g,H] = misfit(m,Q,D,alpha,model)
% Evaluate least-squares misfit
%
%   0.5||P^TA^{-1}(m)Q - D||_{F}^2 + 0.5\alpha||Lm||_2^2,
%
% where P, Q encode the receiver and source locations and L is the first-order FD matrix
%
% use:
%   [f,g,H] = misfit(m,Q,D,model)
%
% input:
%   m - squared-slownes [s^2/km^2]
%   Q - source weights (see F.m)
%   D - single-frequency data matrix
%   alpha - regularization parameter
%   model.h - gridspacing in each direction d = [d1, d2];
%   model.n - number of gridpoints in each direction n = [n1, n2]
%   model.f - frequency [Hz].
%   model.{zr,xr} - {z,x} locations of receivers [m] (must coincide with gridpoints)
%   model.{zs,xs} - {z,x} locations of sources [m] (must coincide with gridpoints)
%
%
% output:
%   f - value of misfit
%   g - gradient (vector of size size(m))
%   H - GN Hessian (Spot operator)


%% get matrices
m = m(:);
L = getL(model.h,model.n);

%% forward solve
[Dp,Jp] = F(m,Q,model);

%% compute f
if (~model.data)
    % f = .5*norm(Dp - D)^2 + .5*alpha*norm(L*m)^2;
    f = .5*norm(Dp - D)^2;
else
    f = real( .5*(Dp - D)'*model.G*(Dp - D) ) ;
end
%% compute g
    if nargout > 1
        if (~model.data)
            g = Jp'*(Dp - D) + alpha*(L'*L)*m;
        else
            g = Jp'*(model.G*(Dp - D));
        end

    end
%% get H
    if nargout > 2
        if (~model.data)
            H = Jp'*Jp + alpha*opMatrix(L'*L);
        else
            H = Jp'*(opMatrix(model.G'*model.G)*Jp);
        end
    end
end
