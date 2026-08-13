%% gen_pressure_bc.m
% ============================================================
% Generate the inlet-pressure boundary condition.
%
% Author: Rino M. Albertin
% Date:   2026-08-13
%
% DESCRIPTION
%   Constructs a low-dimensional pressure profile on the minimum-y boundary
%   from a sampled mean plus sinusoidal, Gaussian-bump, and linear shape terms.
%   The completed profile is clipped at zero and returned as one inlet vector.
%
% MODEL
%   For x_hat normalized from zero to one along the inlet,
%
%       shape(x_hat) = a_sin*sin(2*pi*f_sin*x_hat + phi_sin)
%                    + sum_i (a_gauss/k_gauss)
%                        * exp(-(x_hat - mu_i)^2/(2*sigma_i^2))
%                    + a_lin*(2*x_hat - 1)
%
%       p_inlet(x_hat) = max(0, p_inlet_mean*(1 + shape(x_hat))).
%
%   When k_gauss > 0, centers are mu_i = i/(k_gauss + 1). Their widths are
%
%       sigma_i = max(sigma_gauss*(1 + gauss_jitter*xi_i),
%                     0.05*sigma_gauss),       xi_i ~ N(0,1).
%
%   Dividing a_gauss by k_gauss keeps the sum of the individual bump
%   coefficients equal to a_gauss as the number of centers changes. Setting all
%   three shape amplitudes to zero
%   gives a uniform inlet pressure.
%
% USAGE
%   [fields, info] = gen_pressure_bc(fields, opts)
%
% INPUTS
%   fields
%       Struct containing fields.grid.X and fields.grid.Y.
%   opts.p_inlet_mean
%       Required mean inlet pressure [Pa].
%
% OPTIONAL OPTS
%   a_sin = 0.0
%       Sinusoidal amplitude relative to p_inlet_mean.
%   f_sin = 1.0
%       Sinusoidal frequency across the normalized inlet.
%   phi_sin
%       Sinusoidal phase [rad]; required when a_sin is nonzero.
%   k_gauss = 0
%       Number of equally spaced interior Gaussian centers.
%   a_gauss = 0.0
%       Total Gaussian-bump amplitude relative to p_inlet_mean.
%   sigma_gauss = 0.12
%       Nominal Gaussian width in normalized inlet coordinates.
%   gauss_jitter = 0.25
%       Relative Gaussian width perturbation.
%   a_lin = 0.0
%       End-to-end linear trend amplitude relative to p_inlet_mean.
%   enable_hooks = false, hooks = struct()
%       Optional p_inlet completion callback.
%
% OUTPUTS
%   fields
%       Input struct with fields.bc replaced by a struct containing the
%       1-by-nx fields.bc.p_inlet vector.
%   info
%       Inlet-pressure statistics and the stored generator settings.
%
% NOTES
%   The Gaussian branch consumes k_gauss normal draws whenever k_gauss is
%   positive and a_gauss is nonzero, even when gauss_jitter is zero. These
%   draws intentionally continue the spatial RNG stream initialized by
%   gen_structure_field. No random values are drawn by the other shape terms.
% ============================================================

function [fields, info] = gen_pressure_bc(fields, opts)

%% === Defaults & hooks ======================================
if nargin < 2 || isempty(opts)
    opts = struct();
end

if ~isfield(opts,'enable_hooks'), opts.enable_hooks = false; end
if ~isfield(opts,'hooks'),        opts.hooks = struct(); end

call_hook = @(name,data) ...
    (opts.enable_hooks ...
     && isfield(opts.hooks,name) ...
     && isa(opts.hooks.(name),'function_handle')) ...
     && opts.hooks.(name)(data);

%% === Required parameter ====================================
if ~isfield(opts,'p_inlet_mean')
    error('gen_pressure_bc:MissingMeanPressure', ...
        'opts.p_inlet_mean must be provided (sampled inlet pressure).');
end

%% === Fixed safety bound ====================================
p_min = 0.0;   % FIXED, internal

%% === Defaults (shape parameters) ===========================
if ~isfield(opts,'a_sin'),        opts.a_sin = 0.0; end
if ~isfield(opts,'f_sin'),        opts.f_sin = 1.0; end

if ~isfield(opts,'k_gauss'),      opts.k_gauss = 0; end
if ~isfield(opts,'a_gauss'),      opts.a_gauss = 0.0; end
if ~isfield(opts,'sigma_gauss'),  opts.sigma_gauss = 0.12; end
if ~isfield(opts,'gauss_jitter'), opts.gauss_jitter = 0.25; end

if ~isfield(opts,'a_lin'),        opts.a_lin = 0.0; end

%% === Extract grid ==========================================
X = fields.grid.X;
Y = fields.grid.Y;

nx = size(X,2);

% robust y = 0 detection
y0_mask = abs(Y(:,1) - min(Y(:))) < 1e-12;
if ~any(y0_mask)
    error('gen_pressure_bc:NoY0Boundary', ...
        'No grid row corresponds to y = 0.');
end

x = X(y0_mask, :);
x_hat = (x - min(x)) / max(max(x) - min(x), eps);

%% === Shape construction ====================================
shape = zeros(1, nx);

% --- Sinus --------------------------------------------------
if opts.a_sin ~= 0
    shape = shape + opts.a_sin * ...
        sin(2*pi*opts.f_sin*x_hat + opts.phi_sin);
end

% --- Gaussian bumps -----------------------------------------
if opts.k_gauss > 0 && opts.a_gauss ~= 0

    k = opts.k_gauss;

    % equally spaced centers in (0,1)
    mu = linspace(0, 1, k+2);
    mu = mu(2:end-1);

    % per-bump sigma jitter
    sigma0 = opts.sigma_gauss;
    jitter = opts.gauss_jitter;

    sigma_i = sigma0 * (1 + jitter * randn(1,k));
    sigma_i = max(sigma_i, 0.05 * sigma0);

    % normalized amplitudes
    a_i = opts.a_gauss / max(k,1);

    for i = 1:k
        shape = shape + a_i * ...
            exp(-(x_hat - mu(i)).^2 ./ (2*sigma_i(i)^2));
    end
end

% --- Linear gradient ----------------------------------------
if opts.a_lin ~= 0
    shape = shape + opts.a_lin * (2*x_hat - 1);
end

%% === Final inlet pressure ==================================
p_mean  = opts.p_inlet_mean;
p_inlet = p_mean * (1 + shape);

% safety clipping
p_inlet = max(p_inlet, p_min);

call_hook('p_inlet', struct('p_inlet', p_inlet));

%% === Store in fields =======================================
fields.bc = struct();
fields.bc.p_inlet = p_inlet;

%% === Statistics ============================================
info.statistics.p_inlet.mean = mean(p_inlet);
info.statistics.p_inlet.std  = std(p_inlet);
info.statistics.p_inlet.min  = min(p_inlet);
info.statistics.p_inlet.max  = max(p_inlet);

%% === Metadata ==============================================
info.parameters = struct( ...
    'p_inlet_mean', opts.p_inlet_mean, ...
    'a_sin',        opts.a_sin, ...
    'f_sin',        opts.f_sin, ...
    'k_gauss',      opts.k_gauss, ...
    'a_gauss',      opts.a_gauss, ...
    'sigma_gauss',  opts.sigma_gauss, ...
    'gauss_jitter', opts.gauss_jitter, ...
    'a_lin',        opts.a_lin );

end
