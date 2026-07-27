%% gen_porosity_field.m
% ============================================================
% Generate synthetic 2D porosity fields ε(x,y) for porous media
%
% Author: Rino M. Albertin
% Date:   2026-01-04
%
% DESCRIPTION
%   Generates a physically consistent porosity field ε(x,y) [-]
%   using a minimal, strictly separated pipeline.
%
%   The porosity field is NOT coupled point-wise to the permeability
%   realization. Instead, a global porosity level is anchored to the
%   sampled permeability scale k_mean via a Kozeny–Carman inspired
%   relation. Local spatial variability is introduced purely through
%   geometry.
%
%   Pipeline:
%
%     1) Structural backbone z_bg(x,y) is taken as geometric input
%     2) z_bg is smoothed ONCE to represent macroscopic porosity scale
%     3) A zero-mean, unit-RMS relative texture t(x,y) is derived
%     4) A global reference porosity ε_ref is obtained by inverting a
%        Kozeny–Carman inspired relation using:
%
%            A_mat = A_rel · k_mean
%
%        where k_mean is the sampled permeability level (DoE parameter)
%     5) Final porosity field is constructed as:
%
%            ε(x,y) = ε_ref + Δε · t(x,y)
%
%     6) The field is clipped to global physical bounds and returned
%
%   IMPORTANT DESIGN CHOICES
%   ------------------------
%   • Kozeny–Carman is used ONLY as a global level anchor
%   • No point-wise enforcement between ε and κ
%   • No feedback from the permeability realization κ(x,y)
%   • Porosity variability is purely geometric, not material-driven
%
% ------------------------------------------------------------------------
% USAGE
%   [fields, info] = gen_porosity_field(fields, opts)
%
% INPUTS
%   fields – Struct containing:
%       fields.grid.X              spatial grid
%       fields.structure.z_bg      structural backbone field
%
% OPTIONAL STRUCT opts
%   (GLOBAL PHYSICAL BOUNDS)
%   .eps_min_global   scalar, default 0.30
%       Lower physical bound for porosity
%   .eps_max_global   scalar, default 0.80
%       Upper physical bound for porosity
%   (GEOMETRIC SCALE)
%   .eps_smooth_rel   scalar in [0,1], default 0.025
%       Relative smoothing length applied ONCE to z_bg
%   (TEXTURE)
%   .texture_amp      scalar >= 0, default 0.01
%       Absolute porosity fluctuation amplitude Δε
%   (MATERIAL LEVEL – MUST BE SAMPLED UPSTREAM)
%   .k_mean           scalar > 0
%       Sampled global permeability level [m²]
%   .A_rel            scalar > 0
%       Dimensionless material coupling factor
%       (defines A_mat = A_rel · k_mean)
%
% OUTPUTS
%   fields.material.eps    final porosity field ε(x,y)
% ============================================================


function [fields, info] = gen_porosity_field(fields, opts)

%% === Default options & hooks ================================
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

%% === Defaults ===============================================
% Global physical safety bounds
if ~isfield(opts,'eps_min_global'), opts.eps_min_global = 0.30; end
if ~isfield(opts,'eps_max_global'), opts.eps_max_global = 0.80; end

% Geometric smoothing
if ~isfield(opts,'eps_smooth_rel'), opts.eps_smooth_rel = 0.025; end

% Absolute porosity fluctuation
if ~isfield(opts,'texture_amp'),    opts.texture_amp = 0.01; end

% Material coupling (must be sampled!)
if ~isfield(opts,'A_rel')
    error('gen_porosity_field:MissingA_rel', ...
        'opts.A_rel must be provided (sampled material coupling).');
end

% k_mean must exist (sampled upstream)
if ~isfield(opts,'k_mean')
    error('gen_porosity_field:MissingkMean', ...
        'opts.k_mean must be provided (sampled permeability level).');
end

%% === Extract inputs =========================================
X     = fields.grid.X;
z_bg  = fields.structure.z_bg;

call_hook('eps_input', struct('z_bg', z_bg));

dx = X(1,2) - X(1,1);

eps_min = opts.eps_min_global;
eps_max = opts.eps_max_global;

%% === Smooth backbone (ONCE) =================================
if opts.eps_smooth_rel > 0
    Lx = X(1,end) - X(1,1);
    sigma_eps = max(opts.eps_smooth_rel * Lx / dx, 1.0);

    k = ceil(6 * sigma_eps);
    [xk,yk] = meshgrid(-k:k, -k:k);

    G = exp(-(xk.^2 + yk.^2) / (2 * sigma_eps^2));
    G = G / sum(G(:));

    z_eps = conv2(z_bg, G, 'same');
else
    z_eps = z_bg;
end

call_hook('eps_smoothed', struct('z_eps', z_eps));

%% === Relative porosity texture ==============================
z0 = z_eps - mean(z_eps(:));
z0 = z0 ./ max(std(z0(:)), eps);

t = z0 - mean(z0(:));
t = t ./ max(rms(t(:)), eps);

%% === KC level anchor (GLOBAL, case-wise) ====================
k_mean = opts.k_mean;
A_mat  = opts.A_rel * k_mean;

kc_fun = @(porosity) A_mat * (porosity.^3) ./ max((1 - porosity).^2, eps);

lo = eps_min + 1e-6;
hi = eps_max - 1e-6;

for it = 1:80
    mid = 0.5 * (lo + hi);
    if kc_fun(mid) > k_mean
        hi = mid;
    else
        lo = mid;
    end
end

eps_ref = 0.5 * (lo + hi);

call_hook('eps_level', struct( ...
    'eps_ref', eps_ref, ...
    'k_ref',   k_mean, ...
    'A_mat',   A_mat ));

%% === Final porosity field ===================================
porosity = eps_ref + opts.texture_amp * t;
porosity = min(max(porosity, eps_min), eps_max);

fields.material.eps = porosity;

call_hook('eps_final', struct('eps', porosity));

%% === Statistics =============================================
info.statistics.eps.mean = mean(porosity(:));
info.statistics.eps.std  = std(porosity(:));
info.statistics.eps.min  = min(porosity(:));
info.statistics.eps.max  = max(porosity(:));

%% === Metadata ===============================================
info.parameters = struct( ...
    'eps_min_global', opts.eps_min_global, ...
    'eps_max_global', opts.eps_max_global, ...
    'eps_smooth_rel', opts.eps_smooth_rel, ...
    'texture_amp',    opts.texture_amp, ...
    'A_rel',          opts.A_rel, ...
    'A_mat',          A_mat, ...
    'eps_ref',        eps_ref);

end