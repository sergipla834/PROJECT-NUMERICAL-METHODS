% Fixed spatial resolution
Nx = 800;

% Different numbers of time steps
Nt_values = [2000, 5000, 10000, 20000];

% CPU time arrays
cpu_FE  = zeros(size(Nt_values));
cpu_RK4 = zeros(size(Nt_values));
cpu_CN  = zeros(size(Nt_values));

for i = 1:length(Nt_values)

    Nt = Nt_values(i);
    
    % --- Forward Euler
    C_FE = zeros(Nt, Nx);
    C_FE(1,:) = c0.';
    tStart = tic;
    for n = 1:Nt-1
       
        c = C_FE(n,:).';  % column

        cxx = laplacian_neumann(c, dx);
        reaction = p.rho * c .* (1 - c/p.cmax);

        c_next = c + dt * (p.D*cxx + reaction);

        c_next = max(c_next, 0);

        C_FE(n+1,:) = c_next.';
    end
    cpu_FE(i) = toc(tStart);

    % --- Runge-Kutta 4
    tStart = tic;
    for n = 1:Nt-1
        c = C_FE(n,:).';  % column vector at time t(n)

    % k1 = f(c_n)
        k1 = rhs_MOL_reacdiff(c, p, dx, Nx);

    % k2 = f(c_n + dt/2 * k1)
         k2 = rhs_MOL_reacdiff(c + 0.5*dt*k1, p, dx, Nx);

    % k3 = f(c_n + dt/2 * k2)
        k3 = rhs_MOL_reacdiff(c + 0.5*dt*k2, p, dx, Nx);

    % k4 = f(c_n + dt * k3)
        k4 = rhs_MOL_reacdiff(c + dt*k3, p, dx, Nx);

    % ERK4 update: c_{n+1} = c_n + dt/6*(k1 + 2k2 + 2k3 + k4)
        c_next = c + (dt/6) * (k1 + 2*k2 + 2*k3 + k4);

    
        c_next = max(c_next, 0);

        C_FE(n+1,:) = c_next.';
    end
    cpu_RK4(i) = toc(tStart);

    % --- Crank-Nicolson
    Lmat = laplacian_matrix_neumann(Nx, dx); % sparse
    I = speye(Nx);

    C_CN = zeros(Nt, Nx);
    C_CN(1,:) = c0.';

    newton_tol = 1e-10;
    newton_maxit = 25;
    tStart = tic;
    for n = 1:Nt-1
        c_n = C_CN(n,:).';

    % f(c_n)
    Rn = p.rho * c_n .* (1 - c_n/p.cmax);
    fn = p.D*(Lmat*c_n) + Rn;

    % Predictor for Newton (explicit Euler step)
    c = c_n + dt*fn;

    % We solve: c - c_n - dt/2 * [ f(c_n) + f(c) ] = 0
    % => G(c) = c - c_n - dt/2*( fn + D*L*c + rho*c(1-c/cmax) ) = 0
    for k = 1:newton_maxit
        Rc  = p.rho * c .* (1 - c/p.cmax);
        fc  = p.D*(Lmat*c) + Rc;

        G = c - c_n - (dt/2)*(fn + fc);

        if norm(G, inf) < newton_tol
            break;
        end

        % Jacobian of G:
        % G(c) = c - c_n - dt/2*( fn + D*L*c + R(c) )
        % dG/dc = I - dt/2*( D*L + dR/dc )
        dR = p.rho * (1 - 2*c/p.cmax); % vector
        J = I - (dt/2)*( p.D*Lmat + spdiags(dR,0,Nx,Nx) );

        delta = J \ G;
        c = c - delta;

        if norm(delta, inf) < 1e-12
            break;
        end
    end

    
    c = max(c, 0);

    C_CN(n+1,:) = c.';
    end
    cpu_CN(i) = toc(tStart);

end

% Plot CPU time vs Nt
figure;
plot(Nt_values, cpu_FE,  '-o', 'LineWidth', 1.5); hold on;
plot(Nt_values, cpu_RK4, '-s', 'LineWidth', 1.5);
plot(Nt_values, cpu_CN,  '-^', 'LineWidth', 1.5);

grid on;
xlabel('Number of time steps N_t');
ylabel('Execution time (s)');
legend('Forward Euler', 'RK4', 'Crank-Nicolson', 'Location', 'northwest');
title('Execution time vs number of time steps');

%% ==========================================================
%                 LOCAL FUNCTIONS
%% ==========================================================

function u0 = ic_gaussian(x,p)
u0 = p.A * exp(-(x - p.x0).^2 / (2*p.sigma^2));
end

function [c,f,s] = pdefun_reacdiff(x,t,u,dudx,p) %#ok<INUSD>
c = 1;
f = p.D * dudx;
s = p.rho * u * (1 - u/p.cmax);
end

function [pl,ql,pr,qr] = bcfun_neumann(xl,ul,xr,ur,t,p) %#ok<INUSD,INUSD,INUSD,INUSD>
pl = 0; ql = 1;
pr = 0; qr = 1;
end

function dcdt = rhs_MOL_reacdiff(c,p,dx,Nx)
cxx = zeros(Nx,1);
cxx(2:Nx-1) = (c(1:Nx-2) - 2*c(2:Nx-1) + c(3:Nx)) / dx^2;
cxx(1)  = 2*(c(2)    - c(1))  / dx^2;
cxx(Nx) = 2*(c(Nx-1) - c(Nx)) / dx^2;

reaction = p.rho * c .* (1 - c/p.cmax);
dcdt = p.D * cxx + reaction;
end

function L = laplacian_matrix_neumann(Nx, dx)
e = ones(Nx,1);
L = spdiags([e -2*e e], -1:1, Nx, Nx) / dx^2;
L(1,1) = -2/dx^2;  L(1,2) = 2/dx^2;
L(Nx,Nx) = -2/dx^2; L(Nx,Nx-1) = 2/dx^2;
end

function cxx = laplacian_neumann(c, dx)
% Same Laplacian used in MOL
Nx = numel(c);
cxx = zeros(Nx,1);
cxx(2:Nx-1) = (c(1:Nx-2) - 2*c(2:Nx-1) + c(3:Nx)) / dx^2;
cxx(1)  = 2*(c(2)    - c(1))  / dx^2;
cxx(Nx) = 2*(c(Nx-1) - c(Nx)) / dx^2;
end