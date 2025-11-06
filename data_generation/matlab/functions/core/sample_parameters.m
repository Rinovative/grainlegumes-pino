%% sample_parameters.m
% ============================================================
% Generate sampled parameter sets for κ(x,y) field generation
% Author: Rino M. Albertin
% Date: 2025-10-16
%
% DESCRIPTION
%   Generates random parameter combinations for synthetic permeability
%   field generation using different sampling strategies (uniform, LHS, Sobol).
%
%   Each generated parameter set can be passed to batch_run.m
%   to create a unique κ(x,y) realization for COMSOL simulation or ML training.
%
% ------------------------------------------------------------------------
% USAGE
%   sample_parameters(method, variation, N, seed, p_log, output_dir)
%
% INPUTS
%   method      – Sampling method: 'uniform', 'lhs', 'sobol'
%   variation   – Relative variation around base values (e.g. 0.1 = ±10 %)
%   N           – Number of parameter sets to generate
%   seed        – Random seed for reproducibility (optional)
%   p_log       – Fraction of lognormal cases (e.g. 0.5 = 50 %) (optional)
%   output_dir  – Optional absolute output directory for saving results
%
% OUTPUTS
%   Saves .csv and .json in output_dir
%
% ============================================================

function sample_parameters(method, variation, N, seed, p_log, output_dir)

%% --- Defaults ----------------------------------------------------------
if nargin < 1, method = 'lhs'; end
if nargin < 2, variation = 0.10; end
if nargin < 3, N = 200; end
if nargin < 4, seed = 42; end
if nargin < 5, p_log = 0.5; end
if nargin < 6
    % Default: go 3 levels up and place under data/meta
    this_file  = mfilename('fullpath');
    script_dir = fileparts(this_file);
    project_root = fullfile(script_dir, '..', '..');
    project_root = char(java.io.File(project_root).getCanonicalPath());
    output_dir = fullfile(project_root, 'data', 'meta');
end

if ~isfolder(output_dir), mkdir(output_dir); end
rng(seed);

valid_methods = ["uniform","lhs","sobol"];
assert(any(strcmpi(method, valid_methods)), ...
    'Invalid method. Use ''uniform'', ''lhs'', or ''sobol''.');

%% --- Base parameters ---------------------------------------------------
base = struct( ...
    'k_mean',          5e-9, ...
    'var_rel',         0.5, ...
    'corr_len_rel',    0.05, ...
    'anisotropy',      [3.0, 1.0], ...
    'volume_fraction', 1.0, ...
    'ms_weight',       [0.7, 0.3], ...
    'ms_scale',        0.1, ...
    'coupling',        0.5 ...
);

param_names = ["k_mean","var_rel","corr_len_rel","anisotropy_x","anisotropy_y", ...
               "volume_fraction","ms_weight_c","ms_weight_f", ...
               "ms_scale","coupling"];
n_params = numel(param_names);

%% --- Sampling setup ----------------------------------------------------
switch lower(method)
    case 'uniform'
        X = rand(N, n_params);
    case 'lhs'
        X = lhsdesign(N, n_params, 'Criterion','maximin','Iterations',50);
    case 'sobol'
        p = sobolset(n_params, 'Skip', 1000, 'Leap', 200);
        p = scramble(p,'MatousekAffineOwen');
        X = net(p, N);
end

%% --- Apply variation ---------------------------------------------------
k_mean         = base.k_mean      .* (1 + variation*(2*X(:,1)-1));
var_rel        = base.var_rel     .* (1 + variation*(2*X(:,2)-1));
corr_len_rel   = base.corr_len_rel.* (1 + variation*(2*X(:,3)-1));
anisotropy_x   = base.anisotropy(1).* (1 + variation*(2*X(:,4)-1));
anisotropy_y   = base.anisotropy(2).* (1 + variation*(2*X(:,5)-1));
volume_fraction= base.volume_fraction .* (1 + variation*(2*X(:,6)-1));
ms_weight_c    = base.ms_weight(1) .* (1 + variation*(2*X(:,7)-1));
ms_weight_f    = base.ms_weight(2) .* (1 + variation*(2*X(:,8)-1));
ms_scale       = base.ms_scale .* (1 + variation*(2*X(:,9)-1));
coupling       = base.coupling + variation*(2*X(:,10)-1);

%% --- Physically valid range enforcement -------------------------------
var_rel(var_rel < 0) = 0;
volume_fraction(volume_fraction < 0)   = 0;
volume_fraction(volume_fraction > 1.0) = 1.0;
coupling(coupling < 0) = 0;
coupling(coupling > 1) = 1;
ms_weight_c(ms_weight_c < 0) = 0;
ms_weight_f(ms_weight_f < 0) = 0;

% Normalise ms_weight pair to sum = 1
sum_w = ms_weight_c + ms_weight_f;
ms_weight_c = ms_weight_c ./ sum_w;
ms_weight_f = ms_weight_f ./ sum_w;

%% --- Lognormal flag sampling ------------------------------------------
lognormal = rand(N,1) < p_log;

%% --- Assemble table ----------------------------------------------------
T = table((1:N)', k_mean, var_rel, corr_len_rel, anisotropy_x, anisotropy_y, ...
           volume_fraction, ms_weight_c, ms_weight_f, ms_scale, ...
           coupling, lognormal, ...
           'VariableNames', {'case_id','k_mean','var_rel','corr_len_rel', ...
                             'ani_x','ani_y','vol_frac', ...
                             'msW_c','msW_f','ms_scale','coupling','lognormal'});

%% --- Metadata ----------------------------------------------------------
meta = struct();
meta.method      = method;
meta.variation   = variation;
meta.N           = N;
meta.seed        = seed;
meta.base        = base;
meta.lognormal_p = p_log;
meta.param_names = param_names;
meta.output_dir  = output_dir;
meta.timestamp   = datestr(now,'yyyy-mm-dd HH:MM:SS');

%% --- Output paths ------------------------------------------------------
fname_base = sprintf('samples_%s_var%.0f_N%d', method, variation*100, N);
path_csv  = fullfile(output_dir, [fname_base '.csv']);
path_json = fullfile(output_dir, [fname_base '.json']);

%% --- Export ------------------------------------------------------------
writetable(T, path_csv, 'Delimiter',';');

fid_json = fopen(path_json,'w');
fprintf(fid_json,'%s', jsonencode(struct('meta',meta,'n_cases',N), 'PrettyPrint',true));
fclose(fid_json);

disp("✅ Parameter-Sampling abgeschlossen:");
disp("   → " + path_csv);

end
