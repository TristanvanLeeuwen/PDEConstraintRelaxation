%% setup
clear;clc
% read model, dx = 20 or 50
dx = 20;
v  = dlmread(['marm_' num2str(dx) '.dat']);
% grid
n  = size(v);
h  = dx*[1 1];
z  = (0:n(1)-1)*h(1);
x  = (0:n(2)-1)*h(2);
[zz,xx] = ndgrid(z,x);

%initial model

% initial model

% % % Linearly increasing

% v0 = @(zz,xx)v(1)+0.5e-3*max(zz-350,0);
% m0 = vec(1./v0(zz,xx).^2);

% % % Marmousi
v0 = imgaussfilt (v,22);
m0 = vec(1./v0.^2);
%%
% set frequency, can be an array
% do not set larger than min(1e3*v(:))/(7.5*dx) or smaller than 0.5

f  = [4];

% receivers, xr = .1 - 10km, with 4*dx spacing, zr = 2*dx
xr = 100:4*dx:10000;
zr = 2*dx*ones(1,length(xr));

% sources, xr = .1 - 10km, with 4*dx spacing, zr = 2*dx
xs = 100:4*dx:10000;
zs = 2*dx*ones(1,length(xs));

%% observed data

% regularization parameter
alpha = 1;
% parameters
model.data = 1; %whether use data-driven matrix
model.alpha = alpha;
model.f = f;
model.n = n;
model.h = h;
model.zr = zr;
model.xr = xr;
model.zs = zs;
model.xs = xs;
model.w = f*0+1;
% model
m = 1./v(:).^2;

% source
Q = eye(length(xs));

% data
if model.data
    [D,~,G_data] = F(m,Q,model);
    model.G = G_data;
else 
    D = F(m,Q,model);
end


%% inversion
% misfit

if (~model.data && alpha > 0)
    % % fh = @(m)misfit_pen(m,Q,D,alpha,model);
    fh = @(m) misfit_pen_new(m,D,alpha,model,0);
else
    fh = @(m)misfit(m,Q,D,alpha,model);
end
maxiter = 100;
tol = 1e-6;
c1 = 1e-4;
c2 = 0.9;
x_min = min(m(:));
x_max = max(m(:));
[m_inv,niter, obj_all] = lbfgs_sol(fh,maxiter,tol,10,m0,1,c1,c2,x_min,x_max);
% % % gradient descent algorithm
% [m_inv,niter,obj_all,~] = Descent_BTLS(fh,fh,1,1e-4,0.5,m0,...
               % maxiter,x_min,x_max);

%% plot
vk = reshape(real(1./sqrt(m_inv)),n);
%% plot the inverted velocity

h  = dx*[1 1];
z  = (0:n(1)-1)*h(1);
x  = (0:n(2)-1)*h(2);
x = x/1000;z = z/1000;

% plot ground truth velocity
figure(100);
imagesc(x,z,v,[min(v(:)) max(v(:))]);title('ground truth');colorbar;axis equal tight
xlabel('km');ylabel('km');h = colorbar; 
h.Label.String = 'km/s';
set(gca,'Fontsize',20)

% plot initial velocity
figure; 
imagesc(x,z,v0,[min(v(:)) max(v(:))]);title('initial');axis equal tight

figure;
imagesc(x,z,vk,[min(v(:)) max(v(:))]);title('\alpha=0, final');axis equal tight;colorbar
xlabel('km');ylabel('km');hh = colorbar; 
hh.Label.String = 'km/s';
set(gca,'Fontsize',20)

%% plot the reconstructed data based on the inverted velocity
D_inv = F(m_inv,Q,model);
D_t = F(m,Q,model);
DD = reshape(D_inv, 124,124);
DT = reshape(D_t, 124,124);

diff = reshape(D_t-D_inv,[size(Q),length(f)]);

figure;imagesc(abs(diff(:,:,1)));colorbar
figure;plot(abs(DD(:,42)));hold on;plot(abs(DT(:,42)))
figure;imagesc(abs(reshape(D-D_0,size(Q))));colorbar 