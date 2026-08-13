%% sample_parameters.m
% ============================================================
% Generate reproducible parameter designs and their scientific contract.
%
% Author: Rino M. Albertin
% Date:   2026-08-13
%
% DESCRIPTION
%   Defines the current ordered 28-coordinate design and generates uniform,
%   maximin Latin-hypercube, or scrambled Sobol samples. The same function
%   also exposes the authoritative base values and generation contract without
%   sampling, changing the RNG, or writing files.
%
% SAMPLING METHODS
%   uniform
%       Independent U(0,1) coordinates from MATLAB's seeded random stream.
%   lhs
%       Latin-hypercube design optimized with the maximin criterion over
%       50 iterations.
%   sobol
%       A 28-dimensional Sobol design with Skip = 1000, Leap = 200, and
%       Matousek-Affine-Owen scrambling.
%
%
% SOBOL EXECUTION MASK
%   Sobol CSV files append a logical simulate column. Exactly the first
%   min(1000, N) rows are selected for physical simulation. Uniform and
%   Latin-hypercube files contain no simulate column.
%
% USAGE
%   sample_parameters(method, variation, N, seed, output_dir)
%   [base, param_names, generation_contract] = ...
%       sample_parameters(method, variation, N, seed, output_dir)
%   [base, param_names, generation_contract] = ...
%       sample_parameters('contract', variation)
%
% INPUTS
%   method
%       "uniform", "lhs", or "sobol" in sampling mode; default "lhs".
%       Use "contract" only with variation for a side-effect-free query.
%   variation
%       Finite non-negative design variation; default 0.20.
%   N
%       Number of design rows; default 200.
%   seed
%       Sampling seed; default 42.
%   output_dir
%       Destination for the semicolon-delimited CSV and JSON metadata files.
%       Defaults to the generated-data metadata directory and is created when
%       absent.
%
% OUTPUTS
%   base
%       Base values for every sampled control.
%   param_names
%       Exact ordered 28-coordinate schema used after case_id in the CSV.
%   generation_contract
%       Version-1 Kozeny-Carman reference calibration, natural and batch
%       supports, packing-scatter truncation, and global porosity guards.
%
% FILES WRITTEN IN SAMPLING MODE
%   CSV
%       case_id followed by param_names in exact contract order; Sobol designs
%       append simulate as the final column.
%   JSON
%       Sampling identity, seed, base values, ordered parameter names,
%       generation contract, timestamp, and case count. Sobol metadata also
%       records the selected case count and fraction.
%
% NOTES
%   Sampling mode seeds MATLAB's random stream before constructing the design.
%   Contract-query mode requires exactly two arguments and performs no RNG or
%   file-system operations. Batch admission and field generation consume this
%   contract directly.
% ============================================================

function [base, param_names, generation_contract] = ...
        sample_parameters(method, variation, N, seed, output_dir)

if nargin >= 1 && (ischar(method) || ...
        (isstring(method) && isscalar(method))) && ...
        strcmpi(char(method), 'contract')
    if nargin ~= 2
        error('sample_parameters:ContractQuery', ...
            'Contract queries require exactly method and variation.');
    end
    [base, param_names, generation_contract] = ...
        current_sampling_contract(variation);
    return;
end

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
[base, param_names, generation_contract] = current_sampling_contract(variation);
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

texture_amp = base.texture_amp .* exp(span * Z(:,19));

p_inlet_mean = base.p_inlet_mean .* exp(span * Z(:,20));
sigma_gauss  = base.sigma_gauss  .* exp(span * Z(:,25));
gauss_jitter = base.gauss_jitter .* exp(span * Z(:,26));

% --- logit-space ([0,1]) --------------------------------------
coupling = inv_logit(logit(base.coupling) + span * Z(:,9));

noise_level       = inv_logit(logit(base.noise_level)       + span * Z(:,10));
noise_granularity = inv_logit(logit(base.noise_granularity) + span * Z(:,11));
noise_bias        = inv_logit(logit(base.noise_bias)        + span * Z(:,12));

eps_smooth_rel = inv_logit(logit(base.eps_smooth_rel) + span * Z(:,18));

% --- linear (signed, symmetric) -------------------------------
a_sin   = base.a_sin   .* (1 + variation * Z(:,21));
f_sin   = base.f_sin   .* (1 + variation * Z(:,22));
a_gauss = base.a_gauss .* (1 + variation * Z(:,24));
a_lin   = base.a_lin   .* (1 + variation * Z(:,28));

% --- phase (periodic) -----------------------------------------
phi_sin = mod(base.phi_sin + variation*pi*Z(:,23), 2*pi);

% --- discrete --------------------------------------------------
k_gauss = round( ...
    min(5, max(1, base.k_gauss + round(variation * 3 * Z(:,27)))) ...
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
    eps_smooth_rel, texture_amp, ...
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
meta.generation_contract = generation_contract;
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
end

function [base, param_names, generation_contract] = current_sampling_contract(variation)
% Shared current DOE schema and fixed porosity-packing provenance.

if ~(isnumeric(variation) && isscalar(variation) && isfinite(variation) && variation >= 0)
    error('sample_parameters:InvalidVariation', ...
        'variation must be a finite non-negative scalar.');
end

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
    "eps_smooth_rel","texture_amp", ...
    "p_inlet_mean", ...
    "a_sin","f_sin","phi_sin", ...
    "k_gauss","a_gauss","sigma_gauss","gauss_jitter", ...
    "a_lin" ...
];

generation_contract_version = 1;
kappa_nominal = base.k_mean;
eps_nominal = 0.5;
A_KC_reference = kappa_nominal / kc_response(eps_nominal);
reference_id_variation = 0.8;
natural_kappa_support = kappa_support(kappa_nominal, reference_id_variation);
batch_kappa_support = kappa_support(kappa_nominal, variation);
natural_eps_reference_support = kc_porosity_inverse( ...
    natural_kappa_support ./ A_KC_reference);
batch_eps_reference_support = kc_porosity_inverse( ...
    batch_kappa_support ./ A_KC_reference);
eps_min_global = 0.30;
eps_max_global = 0.80;
if batch_eps_reference_support(1) < eps_min_global || ...
        batch_eps_reference_support(2) > eps_max_global
    error('sample_parameters:InvalidBatchKCSupport', ...
        ['Requested variation %.17g maps kappa support [%.17g, %.17g] ' ...
        'to porosity support [%.17g, %.17g], outside global guards [%.17g, %.17g].'], ...
        variation, batch_kappa_support(1), batch_kappa_support(2), ...
        batch_eps_reference_support(1), batch_eps_reference_support(2), ...
        eps_min_global, eps_max_global);
end

generation_contract = struct( ...
    'generation_contract_version', generation_contract_version, ...
    'kappa_nominal', kappa_nominal, ...
    'eps_nominal', eps_nominal, ...
    'A_KC_reference', A_KC_reference, ...
    'reference_id_variation', reference_id_variation, ...
    'natural_kappa_support', natural_kappa_support, ...
    'natural_eps_reference_support', natural_eps_reference_support, ...
    'batch_kappa_support', batch_kappa_support, ...
    'batch_eps_reference_support', batch_eps_reference_support, ...
    'packing_scatter_truncation_lower', -3, ...
    'packing_scatter_truncation_upper', 3, ...
    'eps_min_global', eps_min_global, ...
    'eps_max_global', eps_max_global ...
);
end

function values = kappa_support(kappa_nominal, variation)
values = [kappa_nominal / (1 + variation), ...
    kappa_nominal * (1 + variation)];
end

function value = kc_response(eps_value)
value = eps_value.^3 ./ (1 - eps_value).^2;
end
