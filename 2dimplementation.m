%% ==========================================================
% 2D Bacterial colony expansion (Fisher-KPP) with Backward Euler
%
% PDE (2D):
%   c_t = D*(c_xx + c_yy) + rho*c*(1 - c/cmax)
% BC: Neumann (no-flux) on all boundaries
% IC: Gaussian inoculum at center -> circular expansion
%
% Time integration: Backward Euler + Newton
% Spatial discretization: finite differences, Laplacian 2D via Kronecker (sparse)
%
% Output:
%   - Heatmap of c(x,y,T) + contour (c = level)
%   - Optional: a few snapshots during time
%% ==========================================================

clear; close all; clc;

%% ---------------- Parameters (same as in 1D) ----------------
L  = 200;      % mm (domain 0..L in x and y)
T  = 30;       % days
D  = 0.1;      % mm^2/day
rho  = 1.0;    % 1/day
cmax = 1.0;

A     = 0.1;   % initial amplitude
sigma = 10;    % mm
x0 = L/2; y0 = L/2;

% Grid (tune for speed)
Nx = 90;       
Ny = 90;
x = linspace(0,L,Nx); dx = x(2)-x(1);
y = linspace(0,L,Ny); dy = y(2)-y(1);
[X,Y] = meshgrid(x,y);

% Time grid (tune for speed)
Nt = 800;      
t = linspace(0,T,Nt); dt = t(2)-t(1);

fprintf('2D grid: Nx=%d Ny=%d (N=%d), dx=%.3e, dy=%.3e\n', Nx, Ny, Nx*Ny, dx, dy);
fprintf('Time: Nt=%d, dt=%.3e\n', Nt, dt);

%% ---------------- Initial condition (Gaussian) ----------------
C0 = A * exp(-((X-x0).^2 + (Y-y0).^2)/(2*sigma^2));  % Ny x Nx
c0 = C0(:);  % vector N x 1, column-major

%% ---------------- Build 2D Laplacian (Neumann) ----------------
Lx = laplacian_1D_neumann(Nx, dx);   % Nx x Nx sparse
Ly = laplacian_1D_neumann(Ny, dy);   % Ny x Ny sparse

Ix = speye(Nx); Iy = speye(Ny);

% Note: with column-major vectorization of C (Ny x Nx), the 2D Laplacian is:
% L2D = kron(Ix, Ly) + kron(Lx, Iy)
L2D = kron(Ix, Ly) + kron(Lx, Iy);   % (Ny*Nx) x (Ny*Nx) sparse

I = speye(Nx*Ny);

%% ---------------- Backward Euler + Newton ----------------
C_BE = zeros(Ny, Nx, 3);   % store a few snapshots for plotting
snap_times = [0, T/2, T];  % days
snap_ids = zeros(size(snap_times));

for k = 1:numel(snap_times)
    [~,snap_ids(k)] = min(abs(t - snap_times(k)));
end

c = c0;  % current state

newton_tol   = 1e-9;
newton_maxit = 20;

for n = 1:Nt-1
    c_n = c;

    % Predictor (explicit Euler) helps Newton
    Rn = rho * c_n .* (1 - c_n/cmax);
    f_n = D*(L2D*c_n) + Rn;
    c = c_n + dt*f_n;

    % Newton to solve:
    % G(c) = c - c_n - dt*( D*L2D*c + rho*c*(1-c/cmax) ) = 0
    for it = 1:newton_maxit
        Rc  = rho * c .* (1 - c/cmax);
        G   = c - c_n - dt*( D*(L2D*c) + Rc );

        if norm(G, inf) < newton_tol
            break;
        end

        dR  = rho * (1 - 2*c/cmax);  % pointwise derivative of reaction
        J   = I - dt*( D*L2D + spdiags(dR, 0, Nx*Ny, Nx*Ny) );

        delta = J \ G;
        c = c - delta;

        if norm(delta, inf) < 1e-11
            break;
        end
    end

    % Optional positivity clamp
    c = max(c, 0);

    % Save snapshots
    if any(n+1 == snap_ids)
        idx = find((n+1) == snap_ids, 1);
        C_BE(:,:,idx) = reshape(c, Ny, Nx);
    end
end

C_final = reshape(c, Ny, Nx);

%% ---------------- Plot: circular expansion (heatmap + contour) ----------------
level = 0.5*cmax;  % contour level to visualize the "front"

figure;
imagesc(x, y, C_final);
set(gca,'YDir','normal');
axis equal tight;
colorbar;
xlabel('x (mm)');
ylabel('y (mm)');
title(sprintf('Backward Euler 2D: c(x,y,T),  T=%.1f days', T));
hold on;
contour(x, y, C_final, [level level], 'k', 'LineWidth', 2);
legend(sprintf('Front contour: c=%.2f', level), 'Location', 'southoutside');
hold off;

%% ---------------- Optional: show snapshots (0, T/2, T) ----------------
figure;
for k = 1:3
    subplot(1,3,k);
    imagesc(x, y, C_BE(:,:,k));
    set(gca,'YDir','normal');
    axis equal tight;
    colorbar;
    title(sprintf('t = %.1f days', snap_times(k)));
    xlabel('x (mm)'); ylabel('y (mm)');
    hold on;
    contour(x, y, C_BE(:,:,k), [level level], 'k', 'LineWidth', 1.5);
    hold off;
end
sgtitle('Circular expansion snapshots (Backward Euler 2D)');

%% ==========================================================
%                 LOCAL FUNCTION
%% ==========================================================
function L = laplacian_1D_neumann(N, dx)
% 1D second derivative with Neumann BC using mirror conditions.
% Interior: (1 -2 1)/dx^2
% Boundaries:
%   cxx(1)  = 2*(c2 - c1)/dx^2
%   cxx(N)  = 2*(c_{N-1} - c_N)/dx^2

e = ones(N,1);
L = spdiags([e -2*e e], -1:1, N, N) / dx^2;

L(1,1) = -2/dx^2;  L(1,2) =  2/dx^2;
L(N,N) = -2/dx^2;  L(N,N-1) = 2/dx^2;
end
