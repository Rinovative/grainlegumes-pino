%% gen_porosity_field.m
% ============================================================
% Generate a KC-anchored textured porosity field for porous media.
%
% Author: Rino M. Albertin
% Date:   2026-08-13
%
% DESCRIPTION
%   Generates a physically bounded porosity field from the shared structural
%   backbone, a fixed Kozeny-Carman calibration, and one reproducible packing
%   realization. Permeability supplies only the global KC trend; the local
%   permeability realization is never coupled point-wise to porosity.
%
%   Pipeline:
%     1. Smooth fields.structure.z_bg once at the macroscopic porosity scale.
%     2. Normalize the smoothed backbone to zero mean and unit RMS texture.
%     3. Invert the fixed KC response at the sampled global k_mean.
%     4. Select the natural-ID support or the appropriate extended batch tail.
%     5. Draw one standard-normal packing value truncated to (-3, 3).
%     6. Scale that draw by one third of the available support margin.
%     7. Add the spatial texture and clip only to the fixed global guards.
%
% MODEL
%   eps_kc_trend = KC_inverse(k_mean / A_KC_reference)
%   packing_scatter_sigma = packing_scatter_margin / 3
%   eps_reference = eps_kc_trend + packing_scatter_sigma * packing_scatter_z
%   eps_unclipped(x,y) = eps_reference + texture_amp * texture(x,y)
%   eps(x,y) = clip(eps_unclipped, eps_min_global, eps_max_global)
%
% USAGE
%   [fields, info] = gen_porosity_field(fields, opts)
%
% INPUTS
%   fields
%       Struct containing fields.grid.X and fields.structure.z_bg.
%   opts
%       Required fields:
%         k_mean
%           Sampled positive global permeability level [m^2].
%         generation_contract
%           Version-1 KC calibration, natural and batch supports, truncation
%           limits, and fixed global porosity guards.
%         packing_scatter_seed
%           Case-specific seed resolved by gen_simulation_inputs.
%       Optional fields:
%         eps_smooth_rel
%           Relative smoothing length in [0, 1], default 0.025.
%         texture_amp
%           Non-negative absolute texture amplitude, default 0.01.
%         eps_min_global, eps_max_global
%           Defaults 0.30 and 0.80 and must equal the generation contract.
%         enable_hooks, hooks
%           Optional diagnostic callbacks for intermediate porosity stages.
%
% OUTPUTS
%   fields
%       Input struct extended with fields.material.eps.
%   info
%       Exact generation-contract fields, resolved packing provenance,
%       porosity parameters, summary statistics, and local clipping fraction.
%
% NOTES
%   Packing scatter is case provenance, not a DOE coordinate. It uses a local
%   mt19937ar RandStream and does not mutate MATLAB's global RNG state.
%   The final global clipping guard is retained as a fail-safe and is reported
%   through info.statistics.eps.local_clipping_fraction.
% ============================================================

function [fields, info] = gen_porosity_field(fields, opts)

if nargin < 2 || isempty(opts)
    opts = struct();
end
if ~isfield(opts, 'enable_hooks'), opts.enable_hooks = false; end
if ~isfield(opts, 'hooks'), opts.hooks = struct(); end
call_hook = @(name, data) (opts.enable_hooks && isfield(opts.hooks, name) ...
    && isa(opts.hooks.(name), 'function_handle')) && opts.hooks.(name)(data);

if ~isfield(opts, 'eps_smooth_rel'), opts.eps_smooth_rel = 0.025; end
if ~isfield(opts, 'texture_amp'), opts.texture_amp = 0.01; end
if ~isfield(opts, 'eps_min_global'), opts.eps_min_global = 0.30; end
if ~isfield(opts, 'eps_max_global'), opts.eps_max_global = 0.80; end
if ~isfield(opts, 'k_mean')
    error('gen_porosity_field:MissingkMean', ...
        'opts.k_mean must be provided as the sampled permeability level.');
end
if ~isfield(opts, 'generation_contract') || ...
        ~isfield(opts, 'packing_scatter_seed')
    error('gen_porosity_field:MissingPackingIdentity', ...
        'generation_contract and packing_scatter_seed must be resolved by case orchestration.');
end

contract = opts.generation_contract;
validate_contract_and_options(contract, opts);

X = fields.grid.X;
z_bg = fields.structure.z_bg;
call_hook('eps_input', struct('z_bg', z_bg));
dx = X(1,2) - X(1,1);

if opts.eps_smooth_rel > 0
    Lx = X(1,end) - X(1,1);
    smoothing_sigma = max(opts.eps_smooth_rel * Lx / dx, 1.0);
    kernel_radius = ceil(6 * smoothing_sigma);
    [xk, yk] = meshgrid(-kernel_radius:kernel_radius, -kernel_radius:kernel_radius);
    G = exp(-(xk.^2 + yk.^2) / (2 * smoothing_sigma^2));
    G = G / sum(G(:));
    z_eps = conv2(z_bg, G, 'same');
else
    z_eps = z_bg;
end
call_hook('eps_smoothed', struct('z_eps', z_eps));

z0 = z_eps - mean(z_eps(:));
z0 = z0 ./ max(std(z0(:)), eps);
texture = z0 - mean(z0(:));
texture = texture ./ max(rms(texture(:)), eps);

k_mean = opts.k_mean;
reference = resolve_packing_reference( ...
    k_mean, contract, opts.packing_scatter_seed);

call_hook('eps_level', struct( ...
    'eps_kc_trend', reference.eps_kc_trend, ...
    'eps_reference', reference.eps_reference, ...
    'packing_scatter_z', reference.packing_scatter_z));

porosity_unclipped = reference.eps_reference + opts.texture_amp * texture;
porosity = min(max(porosity_unclipped, contract.eps_min_global), ...
    contract.eps_max_global);
fields.material.eps = porosity;
call_hook('eps_final', struct('eps', porosity));

info.statistics.eps.mean = mean(porosity(:));
info.statistics.eps.std = std(porosity(:));
info.statistics.eps.min = min(porosity(:));
info.statistics.eps.max = max(porosity(:));
info.statistics.eps.local_clipping_fraction = ...
    mean(porosity_unclipped(:) < contract.eps_min_global | ...
    porosity_unclipped(:) > contract.eps_max_global);

info.parameters = contract;
info.parameters.k_mean = k_mean;
info.parameters.eps_kc_trend = reference.eps_kc_trend;
info.parameters.packing_scatter_seed = reference.packing_scatter_seed;
info.parameters.packing_scatter_z = reference.packing_scatter_z;
info.parameters.packing_scatter_margin = reference.packing_scatter_margin;
info.parameters.packing_scatter_sigma = reference.packing_scatter_sigma;
info.parameters.packing_support_kind = reference.packing_support_kind;
info.parameters.packing_support_lower = reference.packing_support_lower;
info.parameters.packing_support_upper = reference.packing_support_upper;
info.parameters.eps_reference = reference.eps_reference;
info.parameters.eps_smooth_rel = opts.eps_smooth_rel;
info.parameters.texture_amp = opts.texture_amp;
end

function validate_contract_and_options(contract, opts)
if ~(isnumeric(opts.k_mean) && isscalar(opts.k_mean) && isfinite(opts.k_mean) && opts.k_mean > 0)
    error('gen_porosity_field:InvalidkMean', 'opts.k_mean must be finite and positive.');
end
if ~(isnumeric(opts.eps_smooth_rel) && isscalar(opts.eps_smooth_rel) && ...
        isfinite(opts.eps_smooth_rel) && opts.eps_smooth_rel >= 0 && opts.eps_smooth_rel <= 1)
    error('gen_porosity_field:InvalidSmoothing', ...
        'opts.eps_smooth_rel must be in [0, 1].');
end
if ~(isnumeric(opts.texture_amp) && isscalar(opts.texture_amp) && ...
        isfinite(opts.texture_amp) && opts.texture_amp >= 0)
    error('gen_porosity_field:InvalidTextureAmplitude', ...
        'opts.texture_amp must be finite and non-negative.');
end
if opts.eps_min_global ~= contract.eps_min_global || ...
        opts.eps_max_global ~= contract.eps_max_global
    error('gen_porosity_field:GlobalGuardMismatch', ...
        'opts global porosity guards must equal the fixed generation contract.');
end
end

function reference = resolve_packing_reference(k_mean, contract, packing_scatter_seed)
% Resolve KC trend, active support, and deterministic packing scatter.

if ~(isscalar(k_mean) && isfinite(k_mean) && k_mean > 0)
    error('resolve_packing_reference:InvalidKMean', ...
        'k_mean must be finite and positive.');
end
kappa_tolerance = 64 * eps(max(abs(contract.batch_kappa_support)));
if k_mean < contract.batch_kappa_support(1) - kappa_tolerance || ...
        k_mean > contract.batch_kappa_support(2) + kappa_tolerance
    error('resolve_packing_reference:KMeanOutsideBatchSupport', ...
        'k_mean %.17g is outside batch support [%.17g, %.17g].', ...
        k_mean, contract.batch_kappa_support(1), contract.batch_kappa_support(2));
end
eps_kc_trend = kc_porosity_inverse(k_mean / contract.A_KC_reference);
natural_tolerance = 64 * eps(max(abs(contract.natural_kappa_support)));
if k_mean >= contract.natural_kappa_support(1) - natural_tolerance && ...
        k_mean <= contract.natural_kappa_support(2) + natural_tolerance
    lower = contract.natural_eps_reference_support(1);
    upper = contract.natural_eps_reference_support(2);
    kind = 'natural_id';
elseif k_mean < contract.natural_kappa_support(1)
    lower = contract.batch_eps_reference_support(1);
    upper = contract.natural_eps_reference_support(1);
    kind = 'lower_extended_tail';
else
    lower = contract.natural_eps_reference_support(2);
    upper = contract.batch_eps_reference_support(2);
    kind = 'upper_extended_tail';
end
if lower < contract.eps_min_global || upper > contract.eps_max_global || lower > upper
    error('resolve_packing_reference:InvalidSupport', ...
        'Active %s support [%.17g, %.17g] violates global guards [%.17g, %.17g].', ...
        kind, lower, upper, contract.eps_min_global, contract.eps_max_global);
end
porosity_tolerance = 64 * eps(max(abs([lower, upper, eps_kc_trend])));
margin = min(eps_kc_trend - lower, upper - eps_kc_trend);
if margin < -porosity_tolerance
    error('resolve_packing_reference:NegativeMargin', ...
        'KC trend lies outside its active packing support.');
end
if abs(margin) <= porosity_tolerance
    margin = 0;
end
z = draw_packing_scatter(packing_scatter_seed);
sigma = margin / 3;
eps_reference = eps_kc_trend + sigma * z;
if eps_reference < lower - porosity_tolerance || ...
        eps_reference > upper + porosity_tolerance
    error('resolve_packing_reference:ReferenceOutsideSupport', ...
        'Unclipped eps_reference lies outside active packing support.');
end
reference = struct( ...
    'eps_kc_trend', eps_kc_trend, ...
    'packing_scatter_seed', packing_scatter_seed, ...
    'packing_scatter_z', z, ...
    'packing_scatter_margin', margin, ...
    'packing_scatter_sigma', sigma, ...
    'packing_support_kind', kind, ...
    'packing_support_lower', lower, ...
    'packing_support_upper', upper, ...
    'eps_reference', eps_reference ...
);
end

function z = draw_packing_scatter(seed)
% Draw one true truncated-normal packing scatter from a local stream.

stream = RandStream('mt19937ar', 'Seed', seed);
u = rand(stream, 1, 1);
open_probability = eps + (1 - 2 * eps) * u;
z = packing_scatter_quantile(open_probability);
end

function z = packing_scatter_quantile(probability)
% Quantile of N(0,1) truncated strictly to the fixed interval (-3, 3).

if ~(isscalar(probability) && isfinite(probability) && ...
        probability > 0 && probability < 1)
    error('packing_scatter_quantile:InvalidProbability', ...
        'probability must lie strictly inside (0, 1).');
end
lower = -3;
upper = 3;
phi = @(value) 0.5 * (1 + erf(value / sqrt(2)));
truncated_probability = phi(lower) + (phi(upper) - phi(lower)) * probability;
z = sqrt(2) * erfinv(2 * truncated_probability - 1);
if ~(isfinite(z) && z > lower && z < upper)
    error('packing_scatter_quantile:RangeViolation', ...
        'The inverse-CDF draw must lie strictly inside (-3, 3).');
end
end
