%% ==========================================================
% Rungekutta45(explicit) vs Reference solutions
%
% Reference strategy (as in the report):
%  - Spatial reference: pdepe  -> compare c(x,T)
%  - Temporal reference: MOL + ode45 -> compare c(x0,t)
%  - Check pdepe vs MOL+ode45 overlap (error ~ 0)
%
% PDE: c_t = D c_xx + rho c (1 - c/cmax)
% BC:  Neumann (no-flux)
% IC:  Gaussian inoculum centered at x=L/2
%% ==========================================================

clear; close all; clc;

%% ------------------- Parameters ---------------------------
L = 200.0;      % [mm]
T = 30.0;      % [days]
p.L = L;

% Parameters of the equation
p.D    = 0.126; % [mm^2/days^-1]
p.rho  = 1.0; % [days^-1}
p.cmax = 1.0;

% Parameters of the initial gaussian
p.A = 0.1;
p.x0 = p.L / 2;
p.sigma = 10;

Nx = 800;
x  = linspace(0, L, Nx);
dx = x(2) - x(1); % spatial step

Nt = 20000;
t  = linspace(0, T, Nt);
dt = t(2) - t(1); % temporal step 

% Stability check for Forward Euler (diffusion CFL)
dt_max = dx^2/(2*p.D);
fprintf('dx = %.3e, dt = %.3e, dt_max(diffusion) ~ %.3e\n', dx, dt, dt_max);
if dt > dt_max
    warning('Forward Euler may be unstable: dt > dx^2/(2D). Consider increasing Nt.');
end

%% Initial condition (same for all solvers)
c0 = ic_gaussian(x,p).';   % column vector

%% ==========================================================
%% (A) Reference solution in space: PDEPE
%% ==========================================================
m = 0;
sol_pdepe = pdepe(m, ...
    @(x,t,u,dudx) pdefun_reacdiff(x,t,u,dudx,p), ...
    @(x)          ic_gaussian(x,p), ...
    @(xl,ul,xr,ur,t) bcfun_neumann(xl,ul,xr,ur,t,p), ...
    x, t);

C_pdepe = sol_pdepe(:,:,1);   % size Nt x Nx

%% ==========================================================
%% (B) Reference solution in time: MOL + ODE45
%% ==========================================================
rhs = @(t,c) rhs_MOL_reacdiff(c,p,dx,Nx);

opts = odeset('RelTol',1e-8,'AbsTol',1e-10);
[t_ode, C_ode] = ode45(@(tt,cc) rhs(tt,cc), t, c0, opts); %#ok<ASGLU>
% C_ode is Nt x Nx (evaluated exactly at times 't')

%% ==========================================================
%% (C) Check: pdepe vs MOL+ode45 (error should be ~0)
%% ==========================================================
diff_ref = C_pdepe - C_ode;
ref_err_inf = max(abs(diff_ref), [], 'all');
fprintf('Max difference between pdepe and MOL+ode45 (all x,t): %.3e\n', ref_err_inf);

%% ==========================================================
%% (D) ERK45 (Dormand–Prince 4/5) using MOL (adaptive step)
%% ==========================================================
% Embedded explicit Runge–Kutta of order 4/5 (Dormand–Prince) with
% simple adaptive step-size control as described in the report:
%   h_new = control * h * (tolerance / error)^(1/5)
%
% IMPORTANT: To keep the rest of the script unchanged (error evaluation on
% the uniform grid 't'), we integrate each interval [t(n), t(n+1)] with
% adaptive internal sub-steps and store the solution exactly at t(n+1).

C_FE = zeros(Nt, Nx);
C_FE(1,:) = c0.';

tolerance = 1e-6;
control   = 0.9;

% Safety bounds for adaptive step
h_min = 1e-12;        % minimum allowed step
fac_min = 0.2;        % minimum shrink factor
fac_max = 5.0;        % maximum growth factor
max_substeps = 200000; % avoid infinite loops in pathological cases

for n = 1:Nt-1
    tcur    = t(n);
    tnext   = t(n+1);
    y       = C_FE(n,:).';   % column vector state at tcur
    h       = dt;            % initial guess for internal step

    sub = 0;
    while tcur < tnext
        sub = sub + 1;
        if sub > max_substeps
            error('ERK45: too many substeps at n=%d (t=%.5g). Try relaxing tolerance or reducing dt.', n, tcur);
        end

        % Do not step beyond the next output time
        h = min(h, tnext - tcur);

        % --- Dormand–Prince ERK45 stages (Butcher tableau in the report) ---
        k1 = rhs(tcur,               y);
        k2 = rhs(tcur + (1/5)*h,     y + h*( (1/5)*k1 ));
        k3 = rhs(tcur + (3/10)*h,    y + h*( (3/40)*k1 + (9/40)*k2 ));
        k4 = rhs(tcur + (4/5)*h,     y + h*( (44/45)*k1 + (-56/15)*k2 + (32/9)*k3 ));
        k5 = rhs(tcur + (8/9)*h,     y + h*( (19372/6561)*k1 + (-25360/2187)*k2 + (64448/6561)*k3 + (-212/729)*k4 ));
        k6 = rhs(tcur + 1*h,         y + h*( (9017/3168)*k1 + (-355/33)*k2 + (46732/5247)*k3 + (49/176)*k4 + (-5103/18656)*k5 ));
        k7 = rhs(tcur + 1*h,         y + h*( (35/384)*k1 + (500/1113)*k3 + (125/192)*k4 + (-2187/6784)*k5 + (11/84)*k6 ));

        % 5th-order solution (y5) and embedded 4th-order solution (y4)
        y5 = y + h*( (35/384)*k1 + (500/1113)*k3 + (125/192)*k4 + (-2187/6784)*k5 + (11/84)*k6 );
        y4 = y + h*( (5179/57600)*k1 + (7571/16695)*k3 + (393/640)*k4 + (-92097/339200)*k5 + (187/2100)*k6 + (1/40)*k7 );

        % Error estimate (scaled infinity norm)
        err_vec  = y5 - y4;
        scale    = max(1.0, max(abs(y5)));
        err_norm = max(abs(err_vec)) / scale;

        % Accept/reject and step-size update
        if err_norm <= tolerance
            % accept
            tcur = tcur + h;
            y    = y5;

            % Optional safety: avoid negative values due to numerical oscillations
            y = max(y, 0);
        end

        % Step-size adaptation (works for both accepted and rejected steps)
        if err_norm == 0
            fac = fac_max;
        else
            fac = control * (tolerance / err_norm)^(1/5);
            fac = min(fac_max, max(fac_min, fac));
        end

        h = max(h_min, fac * h);
    end

    % store solution at the required output time t(n+1)
    C_FE(n+1,:) = y.';
end

%% (E) ERRORS AND CALCULATION OF NORMS
%%     Reference = MOL + ode45
%% ==========================================================

% ----------------------------------------------------------
% Absolute error point per point
% ----------------------------------------------------------
Err = abs(C_FE - C_ode);     % Nt x Nx

% ----------------------------------------------------------
% (1) Absolute error in SPACE (t = T)
% ----------------------------------------------------------
e_space = Err(end,:);       % 1 x Nx

figure;
plot(x, e_space, 'LineWidth', 2);
grid on; box on;
xlabel('x (mm)');
ylabel('|error(x,T)|');
title('Absolute error in space (t = T)');

% ----------------------------------------------------------
% (2) Error punto a punto en el TIEMPO (x = L/2)
% ----------------------------------------------------------
x0 = L/2;
[~, ix0] = min(abs(x - x0));

e_time = Err(:,ix0);        % Nt x 1

figure;
plot(t, e_time, 'LineWidth', 2);
grid on; box on;
xlabel('t (days)');
ylabel('|error(x0,t)|');
title(sprintf('Absolute error in time (x = %.2f ≈ L/2)', x(ix0)));

% ----------------------------------------------------------
% (3) Calculation of norms (l2, l_inf, local absolute average error,
% relative average error
% ----------------------------------------------------------

% (1) Global L2 error over space-time  (Eq. 18)
err_global_L2 = sqrt( sum(Err(:).^2) * dx * dt );

% (2) Global Linf error over space-time (Eq. 19)
err_global_Linf = max(Err(:));

% (3) Mean absolute local error (pointwise mean)
err_mean_abs = mean(Err(:));

% (4) Mean relative local error (pointwise mean of relative error)
eps_rel = 1e-14;   % avoid division by zero
err_mean_rel = mean( Err(:) ./ (abs(C_ode(:)) + eps_rel) );

fprintf('\n===== ERROR METRICS vs ode45 =====\n');
fprintf('Global error Linf (x,t) : %.4e\n', err_global_Linf);
fprintf('Global error L2   (x,t) : %.4e\n', err_global_L2);
fprintf('Mean relative error      : %.4e\n', err_mean_rel);
fprintf('Mean absolute error      : %.4e\n', err_mean_abs);

% ----------------------------------------------------------
% (4) Representation of the 3D error
% ----------------------------------------------------------

[X,T] = meshgrid(x, t);
figure;
surf(X, T, Err, 'EdgeColor', 'none');
view(45,30);
colormap turbo;
colorbar;

xlabel('x (mm)');
ylabel('t (days)');
zlabel('|error|');
title('Absolute error in space and time');


%% ==========================================================
%                 LOCAL FUNCTIONS
%% ==========================================================

function u0 = ic_gaussian(x,p)
% Gaussian inoculum centered at L/2

u0 = p.A * exp(-(x - p.x0).^2 / (2*p.sigma^2));
end

function [c,f,s] = pdefun_reacdiff(x,t,u,dudx,p) %#ok<INUSD>
% pdepe form: c*u_t = d/dx(f) + s
c = 1;
f = p.D * dudx;
s = p.rho * u * (1 - u/p.cmax);
end

function [pl,ql,pr,qr] = bcfun_neumann(xl,ul,xr,ur,t,p) %#ok<INUSD,INUSD,INUSD,INUSD>
% Neumann no-flux: f = 0 -> p=0, q=1
pl = 0; ql = 1;
pr = 0; qr = 1;
end

function dcdt = rhs_MOL_reacdiff(c,p,dx,Nx)
% MOL semi-discrete RHS: c_t = D*cxx + reaction
cxx = zeros(Nx,1);

% interior
cxx(2:Nx-1) = (c(1:Nx-2) - 2*c(2:Nx-1) + c(3:Nx)) / dx^2;

% Neumann via mirroring
cxx(1)  = 2*(c(2)    - c(1))  / dx^2;
cxx(Nx) = 2*(c(Nx-1) - c(Nx)) / dx^2;

reaction = p.rho * c .* (1 - c/p.cmax);
dcdt = p.D * cxx + reaction;
end

function cxx = laplacian_neumann(c, dx)
% Same Laplacian used in MOL
Nx = numel(c);
cxx = zeros(Nx,1);
cxx(2:Nx-1) = (c(1:Nx-2) - 2*c(2:Nx-1) + c(3:Nx)) / dx^2;
cxx(1)  = 2*(c(2)    - c(1))  / dx^2;
cxx(Nx) = 2*(c(Nx-1) - c(Nx)) / dx^2;
end
