%% gen_structure_field.m
% ============================================================
% Generate a reproducible multiscale structural field for porous media.
%
% Author: Rino M. Albertin
% Date:   2026-08-13
%
% DESCRIPTION
%   Synthesizes a dimensionless two-dimensional latent description of porous
%   material organization. The result captures correlated background regions,
%   anisotropic fine structure, cross-scale dependence, and localized
%   heterogeneities, but carries no permeability, porosity, tensor, or boundary-
%   condition interpretation. Those mappings are owned by later stages.
%
% SCALE ROLES
%   z_base
%       Isotropically correlated backbone controlling broad organization.
%   z_smooth
%       Anisotropically correlated modulation whose white-noise seed is coupled
%       to the base seed.
%   z_bg
%       Direct weighted superposition of independently normalized base and
%       smooth scales.
%   z_noises
%       Mean-zero, RMS-scaled sum of randomly placed elliptical Gaussian
%       perturbations.
%   z
%       Final standardized structure after adding z_bg and z_noises.
%
% ALGORITHM
%   1. Seed MATLAB's global twister stream for the case realization.
%   2. Build a Cartesian grid with round(L/res) + 1 points per direction.
%   3. Filter Gaussian white noise with isotropic base and anisotropic smooth
%      kernels. The smooth seed is
%
%          coupling*z_seed_base + sqrt(1 - coupling^2)*z_uncorr.
%
%   4. Standardize z_base and z_smooth independently, then apply ms_weight.
%   5. Draw a Poisson number of localized ellipses in normalized coordinates;
%      each has random center, scale, aspect ratio, orientation, and sign.
%   6. Center and RMS-normalize their sum, then scale it by noise_level.
%   7. Add the background and localized terms and standardize the final field.
%
% USAGE
%   [fields, info] = gen_structure_field(Lx, Ly, res, seed, opts)
%
% INPUTS
%   Lx, Ly
%       Positive physical domain dimensions [m].
%   res
%       Target Cartesian grid spacing [m].
%   seed
%       Reproducibility seed for the shared spatial RNG stream.
%   opts
%       Optional generator settings listed below.
%
% BACKGROUND OPTS
%   base_len_rel = 0.10
%       Isotropic base correlation length relative to Lx.
%   smooth_len_rel = 0.05
%       Smooth correlation length relative to Lx before anisotropic stretching.
%   ms_weight = [0.4, 0.6]
%       Direct [base, smooth] combination weights.
%   anisotropy = [3, 1]
%       [x, y] stretch factors applied only to the smooth kernel.
%   coupling = 0.5
%       White-noise cross-scale coefficient: zero is independent and one uses
%       the base seed directly.
%
% LOCALIZED-NOISE OPTS
%   noise_level = 0.2
%       RMS amplitude of z_noises and intensity factor for the Poisson count.
%   noise_granularity = 0.5
%       Morphology coordinate from coarse at zero to fine at one. It varies the
%       characteristic ellipse scale from base_len_rel to base_len_rel/10.
%   noise_bias = 0.5
%       Probability that a localized perturbation has positive sign.
%   enable_hooks = false, hooks = struct()
%       Optional callbacks after filtering, background combination, localized-
%       noise scaling, and final structure assembly.
%
% OUTPUTS
%   fields.grid.X, fields.grid.Y
%       Cartesian coordinates with x varying by column and y by row.
%   fields.structure
%       z_base, z_smooth, z_bg, z_noises, and final z arrays.
%   info
%       Resolved seed and generator settings plus structure and localized-noise
%       statistics. The stored RNG state is captured immediately after seeding.
%
% NOTES
%   This function deliberately initializes MATLAB's global RNG. Permeability
%   orientation jitter and inlet Gaussian-width jitter later continue the same
%   stream so a case remains reproducible from one seed and one call order.
% ============================================================

function [fields, info] = gen_structure_field(Lx, Ly, res, seed, opts)

%% === RNG setup ==============================================
rng(seed, 'twister');
rng_state = rng;

%% === Default options & hooks ================================
if nargin < 5
    opts = struct();
end

% --- Hooks ---------------------------------------------------
if ~isfield(opts,'enable_hooks'), opts.enable_hooks = false; end
if ~isfield(opts,'hooks'),        opts.hooks = struct(); end

call_hook = @(name,data) ...
    (opts.enable_hooks ...
     && isfield(opts.hooks,name) ...
     && isa(opts.hooks.(name),'function_handle')) ...
     && opts.hooks.(name)(data);

%% === Background heterogeneity ===============================
if ~isfield(opts,'base_len_rel'),    opts.base_len_rel   = 0.10; end
if ~isfield(opts,'smooth_len_rel'),  opts.smooth_len_rel = 0.05; end
if ~isfield(opts,'ms_weight'),       opts.ms_weight      = [0.4, 0.6]; end
if ~isfield(opts,'anisotropy'),      opts.anisotropy     = [3, 1]; end
if ~isfield(opts,'coupling'),        opts.coupling       = 0.5; end

%% === Localized Gaussian noises ===============================
if ~isfield(opts,'noise_level'),       opts.noise_level = 0.2; end
if ~isfield(opts,'noise_granularity'), opts.noise_granularity = 0.5; end
if ~isfield(opts,'noise_bias'),        opts.noise_bias  = 0.5; end

%% === Initialize info ========================================
info = struct();
info.statistics = struct();
info.statistics.noise = struct('max_abs',0,'l2_norm',0);

%% === Spatial grid ==========================================
dx = res; dy = res;
nx = round(Lx/dx) + 1;
ny = round(Ly/dy) + 1;

x = linspace(0, Lx, nx);
y = linspace(0, Ly, ny);
[X, Y] = meshgrid(x, y);

fields.grid.X = X;
fields.grid.Y = Y;

%% === Correlation kernels ===================================
len_base   = opts.base_len_rel   * Lx;
len_smooth = opts.smooth_len_rel * Lx;

ax = opts.anisotropy(1);
ay = opts.anisotropy(2);

sigma_smooth_x = (len_smooth / sqrt(8*log(2))) / dx * ax;
sigma_smooth_y = (len_smooth / sqrt(8*log(2))) / dy * ay;
sigma_base     = (len_base   / sqrt(8*log(2))) / dx;

ks = ceil(6 * max([sigma_smooth_x, sigma_smooth_y, sigma_base]));
[xk, yk] = meshgrid(-ks:ks, -ks:ks);

G_smooth = exp(-((xk.^2)/(2*sigma_smooth_x^2) + (yk.^2)/(2*sigma_smooth_y^2)));
G_base   = exp(-((xk.^2 + yk.^2)/(2*sigma_base^2)));

G_smooth = G_smooth / sum(G_smooth(:));
G_base   = G_base   / sum(G_base(:));

%% === Multiscale structure field =============================
z_seed_base   = randn(ny,nx);
z_uncorr      = randn(ny,nx);
z_seed_smooth = opts.coupling * z_seed_base ...
              + sqrt(1 - opts.coupling^2) * z_uncorr;

z_base   = conv2(z_seed_base,   G_base,   'same');
z_smooth = conv2(z_seed_smooth, G_smooth, 'same');
call_hook('filtered_fields', struct( ...
    'base_field',   z_base, ...
    'smooth_field', z_smooth ));

z_base   = (z_base   - mean(z_base(:)))   / std(z_base(:));
z_smooth = (z_smooth - mean(z_smooth(:))) / std(z_smooth(:));

z_bg = opts.ms_weight(1)*z_base + opts.ms_weight(2)*z_smooth;
call_hook('structure_field_bg', struct('z_bg', z_bg));

%% === Sub-scale localized noises =============================
z_noises = zeros(size(z_bg));

if opts.noise_level > 0

    level = opts.noise_level;
    chi   = opts.noise_granularity;
    bias  = opts.noise_bias;

    Xn = X / Lx;
    Yn = Y / Ly;

    len0 = opts.base_len_rel;
    sigma_min = len0 / 10;
    sigma_max = len0;
    sigma_char = sigma_min * (sigma_max / sigma_min)^(1 - chi);

    area_dom   = 1.0;
    area_noise = pi * sigma_char^2;

    n_mean   = level * area_dom / max(area_noise, eps);
    n_noises = poissrnd(n_mean);

    noise_field = zeros(size(z_bg));
    s_spread = log(2);

    for i = 1:n_noises

        cx = rand;
        cy = rand;

        sigma  = sigma_char * exp(0.5 * s_spread * randn);
        aspect = exp(s_spread * randn);

        if rand < 0.5
            sx = sigma * aspect;
            sy = sigma;
        else
            sx = sigma;
            sy = sigma * aspect;
        end

        phi = 2*pi*rand;
        c = cos(phi); s_ = sin(phi);

        Xr =  c*(Xn-cx) + s_*(Yn-cy);
        Yr = -s_*(Xn-cx) + c*(Yn-cy);

        sign_amp = (rand < bias)*2 - 1;

        noise_field = noise_field + sign_amp * exp( ...
            -(Xr.^2./(2*sx^2) + Yr.^2./(2*sy^2)) );
    end

    noise_field = noise_field - mean(noise_field(:));
    noise_field = noise_field / max(rms(noise_field(:)), eps);

    z_noises = level * noise_field;
    call_hook('noises_scaled', struct('field', z_noises));
end

%% === Final structure field ==================================
z = z_bg + z_noises;
z = (z - mean(z(:))) / std(z(:));
call_hook('structure_field', struct( ...
    'z', z, 'z_bg', z_bg, 'z_noises', z_noises ));

fields.structure.z_base   = z_base;
fields.structure.z_smooth = z_smooth;
fields.structure.z_bg     = z_bg;
fields.structure.z_noises = z_noises;
fields.structure.z        = z;

%% === Statistics ==============================================
info.statistics.structure = struct();
info.statistics.structure.z.mean = mean(z(:));
info.statistics.structure.z.std  = std(z(:));
info.statistics.structure.z.min  = min(z(:));
info.statistics.structure.z.max  = max(z(:));

info.statistics.structure.z_bg.mean = mean(z_bg(:));
info.statistics.structure.z_bg.std = std(z_bg(:));
info.statistics.structure.z_noises.rms = rms(z_noises(:));

if opts.noise_level > 0
    info.statistics.noise.max_abs = max(abs(z_noises(:)));
    info.statistics.noise.l2_norm = norm(z_noises(:)) / numel(z_noises);
end

%% === Metadata ==============================================
info.parameters = struct( ...
    'seed', seed, ...
    'rng_state', rng_state, ...
    'background', struct( ...
        'base_len_rel',   opts.base_len_rel, ...
        'smooth_len_rel', opts.smooth_len_rel, ...
        'ms_weight',      opts.ms_weight, ...
        'anisotropy',     opts.anisotropy, ...
        'coupling',       opts.coupling ), ...
    'noise', struct( ...
        'level',       opts.noise_level, ...
        'granularity', opts.noise_granularity, ...
        'bias',        opts.noise_bias ) ...
);

end
