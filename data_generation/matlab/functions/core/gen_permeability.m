%% gen_permeability.m
% ============================================================
% Generate synthetic 2D permeability fields κ(x,y) for porous media
% Author: Rino M. Albertin
% Date: 2025-10-16
%
% DESCRIPTION
%   Generates 2D stochastic permeability fields κ(x,y) [m²] representing
%   the heterogeneous structure of porous or granular materials.
%   The generator combines correlated Gaussian random fields to model
%   multiscale heterogeneity and supports optional sparsity, anisotropy,
%   and log-normal transformation.
%
%   The algorithm consists of:
%       1. Creating Gaussian noise fields for coarse and fine scales
%       2. Introducing a correlation (coupling) between scales
%       3. Applying a binary sparsity mask to control active regions
%       4. Smoothing each field using Gaussian correlation kernels
%          – coarse scale: anisotropic
%          – fine scale: isotropic
%       5. Combining both scales by weighted superposition
%       6. Normalizing and mapping to the final permeability field κ(x,y)
%
% ------------------------------------------------------------------------
% USAGE
%   [kappa, X, Y, info] = gen_permeability(Lx, Ly, res, k_mean, ...
%                                          var_rel, corr_len_rel, seed, opts)
%
% INPUTS
%   Lx, Ly        – Domain size [m]
%   res           – Grid spacing [m]
%   k_mean        – Mean permeability [m²]
%   var_rel       – Relative standard deviation (e.g. 0.5 = 50 %)
%   corr_len_rel  – Correlation length of the coarse scale relative to Lx
%   seed          – Random seed for reproducibility
%
% OPTIONAL STRUCT opts
%   .anisotropy      [ax, ay]   Correlation stretch in x/y direction
%   .volume_fraction  scalar    Fraction of active cells (1 = dense field)
%   .ms_weight        [wc, wf]  Weights for coarse/fine scales
%   .ms_scale         scalar    Relative correlation length of fine scale
%   .coupling         scalar    Cross-scale coupling (0–1)
%   .lognormal        bool      true → log-normal κ-field
%   .save             bool      true → export results
%   .save_dir         string    Output directory
%   .file_tag         string    Optional short filename tag (e.g. 'case0001')
%
% DEFAULT PARAMETERS
%   opts.anisotropy      = [1, 1]
%   opts.volume_fraction = 1.0
%   opts.ms_weight       = [1.0, 0.0]
%   opts.ms_scale        = 0.01
%   opts.coupling        = 0.0
%   opts.lognormal       = false
%   opts.save            = true
%   opts.save_dir        = ../../../data/raw/test
%
% OUTPUTS
%   kappa – [ny × nx] permeability field [m²]
%   X, Y  – coordinate grids [m]
%   info  – struct containing geometry, parameters, statistics and paths
% ============================================================

function [kappa, X, Y, info] = gen_permeability(Lx, Ly, res, k_mean, var_rel, corr_len_rel, seed, opts)

%% --- Default options ---------------------------------------------------
if nargin < 8, opts = struct(); end
if ~isfield(opts,'anisotropy'),      opts.anisotropy = [1,1]; end
if ~isfield(opts,'volume_fraction'), opts.volume_fraction = 1.0; end
if ~isfield(opts,'ms_weight'),       opts.ms_weight = [1.0,0.0]; end
if ~isfield(opts,'ms_scale'),        opts.ms_scale = 0.01; end
if ~isfield(opts,'coupling'),        opts.coupling = 0.0; end
if ~isfield(opts,'lognormal'),       opts.lognormal = false; end
if ~isfield(opts,'save'),            opts.save = true; end
if ~isfield(opts,'file_tag'),        opts.file_tag = ''; end
if ~isfield(opts,'save_dir')
    this_dir = fileparts(mfilename('fullpath'));
    opts.save_dir = fullfile(this_dir, '..', '..', '..', 'data', 'raw', 'test');
end

rng(seed);
rng_state = rng;   % store full RNG state for reproducibility

%% --- Defensive checks --------------------------------------------------
assert(all(size(opts.anisotropy)==[1,2]), 'anisotropy must be [ax, ay]');
assert(all(size(opts.ms_weight)==[1,2]), 'ms_weight must be [wc, wf]');
if sum(opts.ms_weight) == 0
    error('ms_weight must not sum to zero.');
end
opts.ms_weight = opts.ms_weight / sum(opts.ms_weight);

%% --- Spatial grid ------------------------------------------------------
dx = res; dy = res;
nx = round(Lx/dx) + 1;
ny = round(Ly/dy) + 1;
x = linspace(0, Lx, nx);
y = linspace(0, Ly, ny);
[X, Y] = meshgrid(x, y);

%% --- Correlation kernels -----------------------------------------------
corr_len_coarse = corr_len_rel * Lx;
corr_len_fine   = opts.ms_scale * Lx;

ax = opts.anisotropy(1);
ay = opts.anisotropy(2);

sigma1_x = (corr_len_coarse / sqrt(8*log(2))) / dx * ax;
sigma1_y = (corr_len_coarse / sqrt(8*log(2))) / dy * ay;
sigma2   = (corr_len_fine   / sqrt(8*log(2))) / dx;

ks = ceil(6 * max([sigma1_x, sigma1_y, sigma2]));
[xk, yk] = meshgrid(-ks:ks, -ks:ks);

G1 = exp(-((xk.^2)/(2*sigma1_x^2) + (yk.^2)/(2*sigma1_y^2)));   % coarse: anisotropic
G2 = exp(-((xk.^2 + yk.^2)/(2*sigma2^2)));                      % fine: isotropic
G1 = G1 / sum(G1(:));
G2 = G2 / sum(G2(:));

%% --- Field generation --------------------------------------------------
z1 = randn(ny, nx);               % coarse-scale noise
z_uncorr = randn(ny, nx);         % independent noise
z2 = opts.coupling * z1 + sqrt(1 - opts.coupling^2) * z_uncorr;

mask = rand(ny, nx) < opts.volume_fraction;
z1 = z1 .* mask;
z2 = z2 .* mask;

z1 = conv2(z1, G1, 'same');
z2 = conv2(z2, G2, 'same');

z1 = (z1 - mean(z1(:))) / std(z1(:));
z2 = (z2 - mean(z2(:))) / std(z2(:));

w = opts.ms_weight;
z = w(1)*z1 + w(2)*z2;
z = (z - mean(z(:))) / std(z(:));

%% --- Mapping to permeability field -------------------------------------
if opts.lognormal
    s = sqrt(log(1 + var_rel^2));
    kappa = k_mean .* exp(s*z - 0.5*s^2);
else
    kappa = k_mean .* (1 + var_rel .* z);
    kappa(kappa < 0.1*k_mean) = 0.1*k_mean;
end

%% --- Statistics --------------------------------------------------------
info = struct();
info.statistics.mean       = mean(kappa(:));
info.statistics.std        = std(kappa(:));
info.statistics.min        = min(kappa(:));
info.statistics.max        = max(kappa(:));
info.statistics.coeff_var  = info.statistics.std / info.statistics.mean;

data = kappa(:);
n = numel(data);
mu = mean(data);
sigma = std(data);
info.statistics.skewness = sum((data - mu).^3) / ((n - 1) * sigma^3);
info.statistics.kurtosis = sum((data - mu).^4) / ((n - 1) * sigma^4) - 3;
info.statistics.quantiles = quantile(kappa(:), [0.1, 0.5, 0.9]);

%% --- Metadata ----------------------------------------------------------
info.geometry = struct( ...
    'Lx', Lx, 'Ly', Ly, ...
    'dx', dx, 'dy', dy, ...
    'nx', nx, 'ny', ny, ...
    'res', res);

info.parameters = struct( ...
    'k_mean', k_mean, ...
    'var_rel', var_rel, ...
    'corr_len_rel', corr_len_rel, ...
    'seed', seed, ...
    'rng_state', rng_state, ...        
    'anisotropy', opts.anisotropy, ...
    'volume_fraction', opts.volume_fraction, ...
    'ms_weight', opts.ms_weight, ...
    'ms_scale', opts.ms_scale, ...
    'coupling', opts.coupling, ...
    'lognormal', opts.lognormal);

info.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');

%% --- File naming -------------------------------------------------------
if ~isempty(opts.file_tag)
    fname_use = char(opts.file_tag);
else
    fname_use = sprintf(['kappa_mean%.1e_var%.2f_corr%.3f_', ...
                         'ani%.1f-%.1f_log%d_vol%.3f_msW%.2f-%.2f_', ...
                         'msS%.3f_coup%.2f_seed%d'], ...
                         k_mean, var_rel, corr_len_rel, ...
                         opts.anisotropy(1), opts.anisotropy(2), opts.lognormal, ...
                         opts.volume_fraction, ...
                         opts.ms_weight(1), opts.ms_weight(2), ...
                         opts.ms_scale, opts.coupling, seed);
end

path_csv  = char(fullfile(opts.save_dir, [fname_use '.csv']));
path_json = char(fullfile(opts.save_dir, [fname_use '.json']));

info.file = struct( ...
    'filename_base', fname_use, ...
    'path_csv', path_csv, ...
    'path_json', path_json);

%% --- Export ------------------------------------------------------------
if opts.save
    if ~exist(opts.save_dir, 'dir'), mkdir(opts.save_dir); end

    % CSV export (COMSOL input)
    writematrix([X(:), Y(:), kappa(:)], path_csv, 'Delimiter',';');

    % JSON export (Python metadata)
    info_json = info;
    info_json.file = struct( ...
        'filename_base', fname_use, ...
        'extension', {'.csv', '.json'});
    fid_json = fopen(path_json, 'w');
    fprintf(fid_json, '%s', jsonencode(info_json, 'PrettyPrint', true));
    fclose(fid_json);
end
end
