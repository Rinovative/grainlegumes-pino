%% sample_parameters.m
% ============================================================
% Generate sampled parameter sets for κ(x,y) field generation
% Author: Rino M. Albertin
% Date: 2025-10-16
%
% DESCRIPTION
%   Generates random parameter combinations for synthetic permeability
%   field generation using different sampling strategies:
%
%       - 'uniform' : purely random sampling
%       - 'lhs'     : Latin Hypercube Sampling (space-filling)
%       - 'sobol'   : Sobol low-discrepancy sequence
%
%   Variation is applied multiplicatively in log-space for positive parameters
%   and in logit-space for [0,1] parameters to maintain stable distributions
%   even for large variation values (e.g. 0.8 or 1.6).
%
%   SPECIAL BEHAVIOR FOR SOBOL:
%   ---------------------------
%   When method == 'sobol', an additional boolean column `simulate` is added
%   to the output table. This column flags which Sobol samples should be
%   physically simulated (e.g. with COMSOL).
%   The first X% of Sobol samples are marked as:
%       simulate == true
%   The remaining samples are marked as:
%       simulate == false
%
%   This enables hybrid workflows:
%       - expensive physics simulations on a Sobol subset
%       - cheap NN forward passes on the full Sobol design
%
%   For 'uniform' and 'lhs', no `simulate` column is added and ALL cases
%   are assumed to be simulated.
%
% ------------------------------------------------------------------------
% USAGE
%   sample_parameters(method, variation, N, seed, p_log, output_dir)
%
% INPUTS
%   method      – Sampling method: 'uniform', 'lhs', 'sobol'
%   variation   – Variation factor: max multiplicative factor = (1 + variation)
%   N           – Number of parameter sets to generate
%   seed        – Random seed for reproducibility
%   output_dir  – Output directory for saving CSV/JSON
%
% OUTPUTS
%   Saves .csv and .json in output_dir
% ============================================================

function sample_parameters(method, variation, N, seed, output_dir)

%% --- Defaults ----------------------------------------------------------
if nargin < 1, method = 'lhs'; end
if nargin < 2, variation = 0.20; end
if nargin < 3, N = 200; end
if nargin < 4, seed = 42; end
if nargin < 5 || isempty(output_dir)
    generated_data_root = getenv('GENERATED_DATA_ROOT');
    if isempty(generated_data_root)
        this_file = mfilename('fullpath');
        script_dir = fileparts(this_file);
        generation_root = fullfile(script_dir, '..', '..', '..');
        generation_root = char(java.io.File(generation_root).getCanonicalPath());
        generated_data_root = fullfile(generation_root, 'data');
    end
    output_dir = fullfile(generated_data_root, 'meta');
end

if ~isfolder(output_dir), mkdir(output_dir); end
rng(seed);

valid_methods = ["uniform","lhs","sobol"];
assert(any(strcmpi(method, valid_methods)), ...
    'Invalid method. Use ''uniform'', ''lhs'', or ''sobol''.');

%% --- Base parameter definition ----------------------------------------
base = struct( ...
    'k_mean',            5e-9, ...
    'var_rel',           0.5, ...
    'base_len_rel',      0.10, ...
    'smooth_len_rel',    0.05, ...
    'ms_weight',         [0.3, 0.7], ...
    'anisotropy',        [3.0, 1.0], ...
    'coupling',          0.5, ...
    'noise_level',       0.2, ...
    'noise_granularity', 0.5, ...
    'noise_bias',        0.5, ...
    'a_max',             2.0, ...
    'a_gamma',           2.0, ...
    'tensor_strength',   1.0, ...
    'theta_jitter',      0.01, ...
    'theta_smooth_rel',  0.1, ...
    'A_rel',             2.0, ...
    'eps_smooth_rel',    0.05, ...
    'texture_amp',       0.005, ...
    'p_inlet_mean',      350, ...
    'a_sin',             0.03, ...
    'f_sin',             0.75, ...
    'phi_sin',           pi, ...
    'k_gauss',           2, ...
    'a_gauss',           0.05, ...
    'sigma_gauss',       0.05, ...
    'gauss_jitter',      0.25, ...
    'a_lin',             0.025 ...
);

param_names = [ ...
    "k_mean","var_rel", ...
    "base_len_rel","smooth_len_rel", ...
    "msW_c","msW_f", ...
    "ani_x","ani_y", ...
    "coupling", ...
    "noise_level","noise_granularity","noise_bias", ...
    "a_max","a_gamma","tensor_strength", ...
    "theta_jitter","theta_smooth_rel", ...
    "A_rel","eps_smooth_rel","texture_amp", ...
    "p_inlet_mean", ...
    "a_sin","f_sin","phi_sin", ...
    "k_gauss","a_gauss","sigma_gauss","gauss_jitter", ...
    "a_lin"
];

n_params = numel(param_names);

%% === Parameter sampling ======================================

% --- Sampling -------------------------------------------------
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

Z    = 2*X - 1;               % [-1, 1]
span = log(1 + variation);    % log-multiplicative span

logit     = @(x) log(x./(1-x));
inv_logit = @(z) 1 ./ (1 + exp(-z));

%% === Apply variations ========================================

% --- log-space (strictly positive) ----------------------------
k_mean         = base.k_mean         .* exp(span * Z(:,1));
var_rel        = base.var_rel        .* exp(span * Z(:,2));

base_len_rel   = base.base_len_rel   .* exp(span * Z(:,3));
smooth_len_rel = base.smooth_len_rel .* exp(span * Z(:,4));

ani_x = base.anisotropy(1) .* exp(span * Z(:,7));
ani_y = base.anisotropy(2) .* exp(span * Z(:,8));

a_max           = base.a_max           .* exp(span * Z(:,13));
a_gamma         = base.a_gamma         .* exp(span * Z(:,14));
tensor_strength = base.tensor_strength .* exp(span * Z(:,15));

theta_jitter     = base.theta_jitter     .* exp(span * Z(:,16));
theta_smooth_rel = base.theta_smooth_rel .* exp(span * Z(:,17));

A_rel       = base.A_rel       .* exp(span * Z(:,18));
texture_amp = base.texture_amp .* exp(span * Z(:,20));

p_inlet_mean = base.p_inlet_mean .* exp(span * Z(:,21));
sigma_gauss  = base.sigma_gauss  .* exp(span * Z(:,26));
gauss_jitter = base.gauss_jitter .* exp(span * Z(:,27));

% --- logit-space ([0,1]) --------------------------------------
coupling = inv_logit(logit(base.coupling) + span * Z(:,9));

noise_level       = inv_logit(logit(base.noise_level)       + span * Z(:,10));
noise_granularity = inv_logit(logit(base.noise_granularity) + span * Z(:,11));
noise_bias        = inv_logit(logit(base.noise_bias)        + span * Z(:,12));

eps_smooth_rel = inv_logit(logit(base.eps_smooth_rel) + span * Z(:,19));

% --- linear (signed, symmetric) -------------------------------
a_sin   = base.a_sin   .* (1 + variation * Z(:,22));
f_sin   = base.f_sin   .* (1 + variation * Z(:,23));
a_gauss = base.a_gauss .* (1 + variation * Z(:,25));
a_lin   = base.a_lin   .* (1 + variation * Z(:,29));

% --- phase (periodic) -----------------------------------------
phi_sin = mod(base.phi_sin + variation*pi*Z(:,24), 2*pi);

% --- discrete --------------------------------------------------
k_gauss = round( ...
    min(5, max(1, base.k_gauss + round(variation * 3 * Z(:,28)))) ...
);

% --- ms-weight (softmax, sum = 1) ------------------------------
w_c = log(base.ms_weight(1)) + span * Z(:,5);
w_f = log(base.ms_weight(2)) + span * Z(:,6);

w = exp([w_c w_f]);
msW_c = w(:,1) ./ sum(w,2);
msW_f = w(:,2) ./ sum(w,2);

%% === Assemble table ============================================
T = table((1:N)', ...
    k_mean, var_rel, ...
    base_len_rel, smooth_len_rel, ...
    msW_c, msW_f, ...
    ani_x, ani_y, ...
    coupling, ...
    noise_level, noise_granularity, noise_bias, ...
    a_max, a_gamma, tensor_strength, ...
    theta_jitter, theta_smooth_rel, ...
    A_rel, eps_smooth_rel, texture_amp, ...
    p_inlet_mean, ...
    a_sin, f_sin, phi_sin, ...
    k_gauss, a_gauss, sigma_gauss, gauss_jitter, ...
    a_lin, ...
    'VariableNames', ['case_id', param_names]);


%% === Sobol simulate flag =======================================
if strcmpi(method,'sobol')
    n_sim = min(1000, N);
    simulate = false(N,1);
    simulate(1:n_sim) = true;
    T.simulate = simulate;
end

%% === Export ====================================================
fname = sprintf('%s_var%.0f_seed%.0f', method, variation*100, seed);

path_csv  = fullfile(output_dir, fname + ".csv");
path_json = fullfile(output_dir, fname + ".json");

% --- CSV ----------------------------------------------------
writetable(T, path_csv, 'Delimiter',';');

% --- JSON metadata ------------------------------------------
meta = struct();
meta.method    = method;
meta.variation = variation;
meta.N         = N;
meta.seed      = seed;
meta.base      = base;
meta.param_names = param_names;
meta.timestamp = datestr(now,'yyyy-mm-dd HH:MM:SS');

if strcmpi(method,'sobol')
    meta.sobol_n_sim = n_sim;
    meta.sobol_simulate_fraction = n_sim / N;
end

fid = fopen(path_json,'w');
fprintf(fid,'%s', jsonencode(struct( ...
    'meta', meta, ...
    'n_cases', N ), 'PrettyPrint', true));
fclose(fid);
