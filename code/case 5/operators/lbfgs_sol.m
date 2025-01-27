function [x,niter,obj_all] = lbfgs_sol(costgrad,maxiter,tol,m,x_0,alpha0,c1,c2,x_min,x_max)
%   lbfgs_sol Convex optimization by L-BFGS method
%   costgrad(x) -> returns the loss function and the gradient
%   maxiter -> maximum number of iterations
%   tol -> tolerance for termination
%   m -> memory
%   x0 -> starting location

if (m) ss = []; ys = []; rhos = [];
else id = eye(size(x_0,1)); end
obj_all = [];
x = x_0;
[f,g] = costgrad(x);
scale =   0.1/max(abs(g));

g = g*scale;
niter = 0;
fprintf('Iteration=%d, f=%g, norm(g)=%g\n',niter,f,norm(g));
obj_all = [obj_all f];
t = alpha0;
r = g;


while (niter<maxiter && norm(g)>tol)
    % descent direction
    p = -r;
    % Back tracking line search
    if (mod(niter,5)==0)
        t = alpha0;
    end
    if (max(abs(p(:)))*t > 0.5) % only for FWI
        t = 0.5/max(abs(p(:))); %0.5 original
    end
    t1 = 0;
    t2 = 0;
    for ils=1:5
        [f_new,g_new] = costgrad(x+t*p);
        g_new = g_new*scale;
        f_wol = f+c1*t*g'*p;
        curv_wol = c2*g'*p;
        curv_new = g_new'*p;
        fprintf('Line search(%d), f_new=%g <= f_wol(%f)=%g; curv_new=%g >= curv_wol=%g\n',ils,f_new,t,f_wol,curv_new,curv_wol);
        if (f_new <= f_wol)
            if (~c2 || curv_new >= curv_wol)
                break;
            else
                t1 = t;
                if (t2 == 0)
                    t = 5*t;
                else
                    t = 0.5*(t1+t2);
                end
            end
        else
            t2 = t;
            t = 0.5*(t1+t2);
        end
    end
    if (f_new>=f & m>=0)
        fprintf('Line search for lbfgs failed! Doing one steepest descent... \n');
        p = -g;
        t = alpha0;
        if (max(abs(p(:)))*t > 0.5) % only for FWI
            t = 0.5/max(abs(p(:)));
        end
        for ils=1:5
            [f_new,g_new] = costgrad(x+t*p);
            f_wol = f+c1*t*g'*p;
            fprintf('BTLS(%d), f_new=%g <= f_wol(%f)=%g; \n',ils,f_new,t,f_wol);
            if (f_new <= f_wol)
                break;
            else
                t = 0.5*t;
            end
        end
    end
    if (f_new>=f)
        fprintf('Line search for steepest descent failed! Exiting... \n');
        return;
    end
    g0 = g;
    g = g_new;
    f = f_new;
    % calculate s_k
    s = t*p;
    % update x_k+1
    x = x + s;
    if (x_min~=x_max)
        x(x<x_min) = x_min;
        x(x>x_max) = x_max;
    end
    % calculate y_k
    y = g - g0;
    % calculate rho_k
    rho = 1/(y'*s);
    % update iteration number
    niter = niter + 1;
    fprintf('Iteration=%d, f=%g, norm(g)=%g \n',niter,f,norm(g));
    obj_all = [obj_all f];
    % update inverse Hessian
    if (m>=0)
        if (m)
            % L-BFGS with memory m
            if (niter > m)
                % discard first element from storage
                ss = [ss(:,2:end),s];
                ys = [ys(:,2:end),y];
                rhos = [rhos(2:end),rho];
            else
                ss = [ss,s];
                ys = [ys,y];
                rhos = [rhos,rho];
            end
            ih0 = 1/(y'*y)/rho;
            if(ih0<0)
                fprintf('Secant equation vialated! ih0=%g\n',ih0);
                ih0 = 1;
            end
            r = ih_update( g,ih0,ys,ss,rhos,min(niter,m) );
        else
            % BFGS
            if (niter==1)
                ih = y'*s/(y'*y)*id;
                %ih = id;
            end
            ih = (id - rho*y*s')'*ih*(id - rho*y*s') + rho*s*s';
            r = ih*g;
        end
    else
        % Steepest descent
        r = g;
    end
end
