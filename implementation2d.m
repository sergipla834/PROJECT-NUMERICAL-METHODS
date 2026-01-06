clear; close all; clc;

L   = 200;     % mm
T   = 30;      % days
D   = 0.1;     
rho = 0.5;     
cmax = 1.0;

A     = 0.1;   
sigma = 10;    
x0 = L/2; y0 = L/2;


Nx = 120;
Ny = 120;

x = linspace(0,L,Nx); dx = x(2)-x(1);
y = linspace(0,L,Ny); dy = y(2)-y(1);
[X,Y] = meshgrid(x,y);

Nt = 700;                    
t  = linspace(0,T,Nt);
dt = t(2)-t(1);

fprintf('2D: Nx=%d Ny=%d -> N=%d | dx=%.3e dy=%.3e\n', Nx, Ny, Nx*Ny, dx, dy);
fprintf('Time: Nt=%d dt=%.3e\n', Nt, dt);

C0 = A * exp(-((X-x0).^2 + (Y-y0).^2)/(2*sigma^2));  
c0 = C0(:); 

figure;
surf(X, Y, C0, 'EdgeColor','none');
view(45,30);
xlabel('x (mm)'); ylabel('y (mm)'); zlabel('c(x,y,0)');
title('Initial condition: Gaussian peak (t = 0)');
colorbar;
zlim([0, A]);              
axis tight;

figure;
imagesc(x, y, C0);
set(gca,'YDir','normal');
axis equal tight;
xlabel('x (mm)'); ylabel('y (mm)');
title('Initial condition: Gaussian peak (heatmap)');
colorbar;

Lx = laplacian_1D_neumann(Nx, dx);   
Ly = laplacian_1D_neumann(Ny, dy); 
Ix = speye(Nx); Iy = speye(Ny);

L2D = kron(Ix, Ly) + kron(Lx, Iy);   
I = speye(Nx*Ny);

c = c0;

newton_tol   = 1e-9;
newton_maxit = 20;

snap_times = [0, T/2, T];
snap_ids = zeros(size(snap_times));
for k = 1:numel(snap_times)
    [~, snap_ids(k)] = min(abs(t - snap_times(k)));
end
C_snap = zeros(Ny, Nx, numel(snap_times));
C_snap(:,:,1) = C0;

for n = 1:Nt-1
    c_n = c;

    Rn  = rho * c_n .* (1 - c_n/cmax);
    fn  = D*(L2D*c_n) + Rn;
    c   = c_n + dt*fn;

    for it = 1:newton_maxit
        Rc = rho * c .* (1 - c/cmax);
        G  = c - c_n - dt*( D*(L2D*c) + Rc );

        if norm(G, inf) < newton_tol
            break;
        end

        dR = rho * (1 - 2*c/cmax);
        J  = I - dt*( D*L2D + spdiags(dR, 0, Nx*Ny, Nx*Ny) );

        delta = J \ G;
        c = c - delta;

        if norm(delta, inf) < 1e-11
            break;
        end
    end

    
    c = max(c, 0);

    idx = find((n+1) == snap_ids, 1);
    if ~isempty(idx)
        C_snap(:,:,idx) = reshape(c, Ny, Nx);
    end
end

C_final = reshape(c, Ny, Nx);

figure;
surf(X, Y, C_final, 'EdgeColor','none');
view(45,30);
xlabel('x (mm)'); ylabel('y (mm)'); zlabel('c(x,y,T)');
title(sprintf('Backward Euler 2D: final state (t = %.1f days)', T));
colorbar;
zlim([0, 1.05*cmax]);      
axis tight;

level = 0.5*cmax;
figure;
imagesc(x, y, C_final);
set(gca,'YDir','normal');
axis equal tight;
xlabel('x (mm)'); ylabel('y (mm)');
title(sprintf('Backward Euler 2D: circular expansion at t = %.1f days', T));
colorbar;
hold on;
contour(x, y, C_final, [level level], 'k', 'LineWidth', 2);
hold off;

figure;
for k = 1:numel(snap_times)
    subplot(1,3,k);
    surf(X, Y, C_snap(:,:,k), 'EdgeColor','none');
    view(45,30);
    title(sprintf('t = %.1f days', snap_times(k)));
    xlabel('x (mm)'); ylabel('y (mm)'); zlabel('c');
    colorbar;
    if k == 1
        zlim([0, A]);          
    else
        zlim([0, 1.05*cmax]);  
    end
    axis tight;
end
sgtitle('Evolution from Gaussian peak to circular expansion (Backward Euler 2D)');


%                 LOCAL FUNCTION
function L = laplacian_1D_neumann(N, dx)

e = ones(N,1);
L = spdiags([e -2*e e], -1:1, N, N) / dx^2;

L(1,1) = -2/dx^2;  L(1,2) =  2/dx^2;
L(N,N) = -2/dx^2;  L(N,N-1) = 2/dx^2;
end
