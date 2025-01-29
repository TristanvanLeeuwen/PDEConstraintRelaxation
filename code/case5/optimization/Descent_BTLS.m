function [ x,niter,obj_all,x_all ] = Descent_BTLS( costgrad, costobj, t0, alpha,beta,x0,...
               maxiter,  x_min, x_max)
% Steepest Descent method with backtracking line search using quadratic norm 

niter = 0;
x = x0;
obj_all = zeros(maxiter+1,1);
x_all = zeros(maxiter+1,length(x));
while (niter <= maxiter)
    [fx,g] = costgrad(x);
    d = -g;
    obj_all(niter+1) = fx;
    x_all(niter+1,:) = x';

%     if mod(niter, 20) == 0
    fprintf('Iteration=%d, fx=%g\n',niter,fx);
%     end
%     save_name = sprintf(str,niter);
%     save(save_name, 'x');
    
    if (niter==0)
        scale_g = 1/max(abs(g(:)));
        scale_d = 1/max(abs(d(:)));
    end
    
    g = g*scale_g;
    d = d*scale_d;
    
    t = t0;
    
    [fx_new] = costobj(x+t*d);
    counter = 1;
    while (fx_new>fx+alpha*t*sum(g.*d))
        t = beta*t;
        [fx_new] = costobj(x+t*d);
        fprintf('Line search, fx_new=%g, target(%f)=%g\n',fx_new,t,fx+alpha*t*sum(g.*d));
        counter = counter + 1;
        if (counter>20)
            return;
        end
    end
    x = x + t*d;
    niter = niter + 1;
    if (x_min~=x_max)
        x(x<x_min) = x_min;
        x(x>x_max) = x_max;
    end

 end

end