%% test_gen_simulation_inputs_smoke.m
% ============================================================
% Verify the generator pipeline and version-1 scientific contract.
%
% Author: Rino M. Albertin
% Date:   2026-08-13
%
% DESCRIPTION
%   Exercises deterministic field generation, KC calibration and supports,
%   packing-seed semantics, porosity provenance, tensor validity, boundary
%   fields, exact metadata, and the canonical seven-column export.
%
% REQUIREMENTS
%   MATLAB Unit Test framework and generator toolboxes. COMSOL is not required.
% ============================================================

function tests = test_gen_simulation_inputs_smoke

tests = functiontests(localfunctions);
end

function testCanonicalGeneratorContract(testCase)
this_dir = fileparts(mfilename('fullpath'));
core_dir = fullfile(fileparts(this_dir), 'core');
original_path = path;
path_cleanup = onCleanup(@() path(original_path)); %#ok<NASGU>
addpath(genpath(core_dir));

output_dir = tempname;
mkdir(output_dir);
output_cleanup = onCleanup(@() cleanup_temp_directory(output_dir)); %#ok<NASGU>

Lx = 0.12;
Ly = 0.075;
res = 0.015;
seed = 3002;
expected_size = [round(Ly / res) + 1, round(Lx / res) + 1];

opts = smoke_options(output_dir, "smoke_a");
[fields_a, info_a] = gen_simulation_inputs(Lx, Ly, res, seed, opts);
opts.file_tag = "smoke_b";
[fields_b, info_b] = gen_simulation_inputs(Lx, Ly, res, seed, opts);

verifyEqual(testCase, size(fields_a.grid.X), expected_size);
verifyEqual(testCase, size(fields_a.grid.Y), expected_size);
verifyTrue(testCase, all(isfinite(fields_a.grid.X), 'all'));
verifyTrue(testCase, all(isfinite(fields_a.grid.Y), 'all'));

verifyTrue(testCase, isfield(fields_a, 'material'));
verifyTrue(testCase, isfield(fields_a.material, 'K'));
verifyTrue(testCase, isfield(fields_a.material, 'eps'));
K = fields_a.material.K;
verifyTrue(testCase, all(isfield(K, {'Kxx', 'Kxy', 'Kyy'})));
verifyEqual(testCase, size(K.Kxx), expected_size);
verifyEqual(testCase, size(K.Kxy), expected_size);
verifyEqual(testCase, size(K.Kyy), expected_size);
verifyTrue(testCase, all(isfinite(K.Kxx), 'all'));
verifyTrue(testCase, all(isfinite(K.Kxy), 'all'));
verifyTrue(testCase, all(isfinite(K.Kyy), 'all'));
verifyGreaterThan(testCase, min(K.Kxx(:) .* K.Kyy(:) - K.Kxy(:).^2), 0);

porosity = fields_a.material.eps;
verifyEqual(testCase, size(porosity), expected_size);
verifyTrue(testCase, all(isfinite(porosity), 'all'));
verifyGreaterThanOrEqual(testCase, min(porosity(:)), 0.30);
verifyLessThanOrEqual(testCase, max(porosity(:)), 0.80);
verifyGreaterThan(testCase, min(porosity(:)), eps);
verifyFalse(testCase, any(porosity(:) == eps));
verifyEqual(testCase, info_a.porosity.parameters.eps_min_global, 0.30);
verifyEqual(testCase, info_a.porosity.parameters.eps_max_global, 0.80);

porosity_parameters = info_a.porosity.parameters;
porosity_ref = porosity_parameters.eps_reference;
kc_at_trend = porosity_parameters.A_KC_reference * ...
    porosity_parameters.eps_kc_trend^3 / (1 - porosity_parameters.eps_kc_trend)^2;
verifyEqual(testCase, kc_at_trend, opts.k_mean, 'RelTol', 1e-12);
verifyEqual(testCase, porosity_ref, porosity_parameters.eps_kc_trend + ...
    porosity_parameters.packing_scatter_sigma * ...
    porosity_parameters.packing_scatter_z, 'AbsTol', 1e-14);
verifyGreaterThan(testCase, porosity_parameters.packing_scatter_z, -3);
verifyLessThan(testCase, porosity_parameters.packing_scatter_z, 3);
verifyTrue(testCase, isfield(info_a.porosity.statistics.eps, 'local_clipping_fraction'));

verifyTrue(testCase, isfield(fields_a, 'bc'));
verifyTrue(testCase, isfield(fields_a.bc, 'p_inlet'));
verifyEqual(testCase, size(fields_a.bc.p_inlet), [1, expected_size(2)]);
verifyTrue(testCase, all(isfinite(fields_a.bc.p_inlet), 'all'));

canonical_columns = ["x", "y", "Kxx", "Kxy", "Kyy", "eps", "p_bc"];
verifyEqual(testCase, string(info_a.export.export.columns), canonical_columns);
metadata = jsondecode(fileread(info_a.export.paths.json));
verifyEqual(testCase, string(metadata.export.columns(:))', canonical_columns);
verifyTrue(testCase, metadata.fields_present.porosity);
verifyTrue(testCase, isfield(metadata.generator, 'porosity'));
verifyTrue(testCase, isfield(metadata.generator.porosity.statistics, 'eps'));
verifyEqual(testCase, ...
    sort(fieldnames(metadata.generator.porosity.parameters)), ...
    sort(fieldnames(porosity_parameters)));
verifyEqual(testCase, ...
    metadata.generator.porosity.parameters.packing_scatter_seed, ...
    porosity_parameters.packing_scatter_seed);
verifyEqual(testCase, ...
    metadata.generator.porosity.parameters.eps_reference, ...
    porosity_parameters.eps_reference);

raw_a = readmatrix(info_a.export.paths.csv, 'Delimiter', ';');
raw_b = readmatrix(info_b.export.paths.csv, 'Delimiter', ';');
verifyEqual(testCase, size(raw_a), [prod(expected_size), numel(canonical_columns)]);
verifyTrue(testCase, all(isfinite(raw_a), 'all'));
verifyEqual(testCase, raw_a(:, 1), fields_a.grid.X(:));
verifyEqual(testCase, raw_a(:, 2), fields_a.grid.Y(:));
verifyEqual(testCase, raw_a(:, 3), K.Kxx(:));
verifyEqual(testCase, raw_a(:, 4), K.Kxy(:));
verifyEqual(testCase, raw_a(:, 5), K.Kyy(:));
verifyEqual(testCase, raw_a(:, 6), porosity(:));

p_bc = zeros(expected_size);
p_bc(1, :) = fields_a.bc.p_inlet;
verifyEqual(testCase, raw_a(:, 7), p_bc(:));
verifyTrue(testCase, all(isfinite(p_bc), 'all'));

verifyEqual(testCase, fields_b.grid.X, fields_a.grid.X);
verifyEqual(testCase, fields_b.grid.Y, fields_a.grid.Y);
verifyEqual(testCase, fields_b.material.K.Kxx, K.Kxx);
verifyEqual(testCase, fields_b.material.K.Kxy, K.Kxy);
verifyEqual(testCase, fields_b.material.K.Kyy, K.Kyy);
verifyEqual(testCase, fields_b.material.eps, porosity);
verifyEqual(testCase, fields_b.bc.p_inlet, fields_a.bc.p_inlet);
verifyEqual(testCase, raw_b, raw_a);
end

function testPackingScatterContract(testCase)
this_dir = fileparts(mfilename('fullpath'));
core_dir = fullfile(fileparts(this_dir), 'core');
original_path = path;
path_cleanup = onCleanup(@() path(original_path)); %#ok<NASGU>
addpath(genpath(core_dir));

[~, ~, contract] = sample_parameters('contract', 1.2);
verifyEqual(testCase, contract.generation_contract_version, 1);
verifyEqual(testCase, contract.kappa_nominal, 5e-9);
verifyEqual(testCase, contract.eps_nominal, 0.5);
verifyGreaterThan(testCase, contract.A_KC_reference, 0);
verifyEqual(testCase, contract.A_KC_reference, 1e-8, 'RelTol', 1e-14);
verifyEqual(testCase, contract.reference_id_variation, 0.8);
verifyEqual(testCase, contract.natural_kappa_support, ...
    [2.777777777777778e-9, 9e-9], 'RelTol', 1e-14);
verifyEqual(testCase, contract.natural_eps_reference_support, ...
    [0.4421552148025800, 0.5592056257488011], 'AbsTol', 1e-14);
verifyEqual(testCase, contract.batch_kappa_support, ...
    [2.272727272727273e-9, 1.1e-8], 'RelTol', 1e-14);
verifyEqual(testCase, contract.batch_eps_reference_support, ...
    [0.4229733612891449, 0.5794453772514555], 'AbsTol', 1e-14);
verifyEqual(testCase, truncated_standard_normal_quantile(0.25), ...
    -truncated_standard_normal_quantile(0.75), 'AbsTol', 1e-14);

opts = smoke_options(tempdir, "unused");
opts.save = false;
opts.packing_batch_variation = 1.2;
opts.packing_batch_seed = 3001;
opts.packing_case_id = 7;
spatial_seed = 9101;
[fields_natural, natural_info] = ...
    gen_simulation_inputs(0.12, 0.075, 0.015, spatial_seed, opts);
[fields_repeat, repeat_info] = ...
    gen_simulation_inputs(0.12, 0.075, 0.015, spatial_seed, opts);
natural = natural_info.porosity.parameters;
repeat = repeat_info.porosity.parameters;
verifyEqual(testCase, natural.packing_scatter_seed, 1367922492);
verifyEqual(testCase, natural.packing_scatter_seed, repeat.packing_scatter_seed);
verifyEqual(testCase, natural.packing_scatter_z, repeat.packing_scatter_z);
verifyEqual(testCase, natural.packing_scatter_z, ...
    expected_packing_scatter(natural.packing_scatter_seed));
verifyGreaterThan(testCase, natural.packing_scatter_z, -3);
verifyLessThan(testCase, natural.packing_scatter_z, 3);
verifyEqual(testCase, fields_repeat.material.eps, fields_natural.material.eps);

opts.packing_case_id = 8;
[fields_other_scatter, other_info] = ...
    gen_simulation_inputs(0.12, 0.075, 0.015, spatial_seed, opts);
other = other_info.porosity.parameters;
verifyEqual(testCase, other.packing_scatter_seed, 1367922493);
verifyNotEqual(testCase, natural.packing_scatter_seed, other.packing_scatter_seed);
verifyNotEqual(testCase, natural.packing_scatter_z, other.packing_scatter_z);
verifyEqual(testCase, fields_other_scatter.material.K.Kxx, ...
    fields_natural.material.K.Kxx);
verifyEqual(testCase, fields_other_scatter.material.K.Kxy, ...
    fields_natural.material.K.Kxy);
verifyEqual(testCase, fields_other_scatter.material.K.Kyy, ...
    fields_natural.material.K.Kyy);
verifyEqual(testCase, fields_other_scatter.bc.p_inlet, fields_natural.bc.p_inlet);

opts.packing_case_id = 7;
opts.k_mean = 4e-9;
[~, lower_trend_info] = ...
    gen_simulation_inputs(0.12, 0.075, 0.015, spatial_seed, opts);
opts.k_mean = 6e-9;
[~, upper_trend_info] = ...
    gen_simulation_inputs(0.12, 0.075, 0.015, spatial_seed, opts);
verifyLessThan(testCase, ...
    lower_trend_info.porosity.parameters.eps_kc_trend, ...
    upper_trend_info.porosity.parameters.eps_kc_trend);

opts.k_mean = mean([contract.batch_kappa_support(1), ...
    contract.natural_kappa_support(1)]);
[~, lower_tail_info] = ...
    gen_simulation_inputs(0.12, 0.075, 0.015, spatial_seed, opts);
opts.k_mean = mean([contract.natural_kappa_support(2), ...
    contract.batch_kappa_support(2)]);
[~, upper_tail_info] = ...
    gen_simulation_inputs(0.12, 0.075, 0.015, spatial_seed, opts);
lower_tail = lower_tail_info.porosity.parameters;
upper_tail = upper_tail_info.porosity.parameters;
verifyEqual(testCase, natural.packing_support_kind, 'natural_id');
verifyEqual(testCase, lower_tail.packing_support_kind, 'lower_extended_tail');
verifyEqual(testCase, upper_tail.packing_support_kind, 'upper_extended_tail');
verifyLessThan(testCase, lower_tail.eps_reference, ...
    natural.packing_support_lower);
verifyGreaterThan(testCase, upper_tail.eps_reference, ...
    natural.packing_support_upper);
verifyGreaterThan(testCase, natural.eps_reference, ...
    natural.packing_support_lower);
verifyLessThan(testCase, natural.eps_reference, ...
    natural.packing_support_upper);
verifyEqual(testCase, natural.eps_reference, natural.eps_kc_trend + ...
    natural.packing_scatter_sigma * natural.packing_scatter_z, ...
    'AbsTol', 1e-14);

opts.k_mean = contract.natural_kappa_support(1);
[~, boundary_info] = ...
    gen_simulation_inputs(0.12, 0.075, 0.015, spatial_seed, opts);
boundary = boundary_info.porosity.parameters;
verifyEqual(testCase, boundary.packing_scatter_margin, 0, 'AbsTol', 1e-13);
verifyEqual(testCase, boundary.packing_scatter_sigma, 0, 'AbsTol', 1e-13);
verifyEqual(testCase, boundary.eps_reference, boundary.eps_kc_trend, ...
    'AbsTol', 1e-13);

expected_parameter_fields = sort({ ...
    'generation_contract_version'; 'kappa_nominal'; 'eps_nominal'; ...
    'A_KC_reference'; 'reference_id_variation'; 'natural_kappa_support'; ...
    'natural_eps_reference_support'; 'batch_kappa_support'; ...
    'batch_eps_reference_support'; 'packing_scatter_truncation_lower'; ...
    'packing_scatter_truncation_upper'; 'eps_min_global'; 'eps_max_global'; ...
    'k_mean'; 'eps_kc_trend'; 'packing_scatter_seed'; ...
    'packing_scatter_z'; 'packing_scatter_margin'; 'packing_scatter_sigma'; ...
    'packing_support_kind'; 'packing_support_lower'; ...
    'packing_support_upper'; 'eps_reference'; 'eps_smooth_rel'; 'texture_amp'});
verifyEqual(testCase, sort(fieldnames(natural)), expected_parameter_fields);

invalid_opts = opts;
invalid_opts.packing_batch_variation = 100;
rng(99);
expected_next_global_draw = rand;
rng(99);
verifyError(testCase, @() gen_simulation_inputs( ...
    0.12, 0.075, 0.015, spatial_seed, invalid_opts), ...
    'sample_parameters:InvalidBatchKCSupport');
observed_next_global_draw = rand;
verifyEqual(testCase, observed_next_global_draw, expected_next_global_draw);
end


function testStandaloneSeedDefaultsPackingIdentity(testCase)
this_dir = fileparts(mfilename('fullpath'));
core_dir = fullfile(fileparts(this_dir), 'core');
original_path = path;
path_cleanup = onCleanup(@() path(original_path)); %#ok<NASGU>
addpath(genpath(core_dir));

opts = smoke_options(tempdir, "unused");
opts = rmfield(opts, {'packing_batch_seed', 'packing_case_id'});
opts.save = false;
[~, first] = gen_simulation_inputs(0.12, 0.075, 0.015, 3001, opts);
[~, second] = gen_simulation_inputs(0.12, 0.075, 0.015, 3002, opts);
verifyEqual(testCase, first.porosity.parameters.packing_scatter_seed, ...
    1367922486);
verifyEqual(testCase, second.porosity.parameters.packing_scatter_seed, ...
    499337171);
verifyEqual(testCase, first.porosity.parameters.packing_scatter_z, ...
    expected_packing_scatter(1367922486));
verifyEqual(testCase, second.porosity.parameters.packing_scatter_z, ...
    expected_packing_scatter(499337171));
verifyNotEqual(testCase, first.porosity.parameters.packing_scatter_seed, ...
    second.porosity.parameters.packing_scatter_seed);
end

function opts = smoke_options(output_dir, file_tag)
opts = struct( ...
    'base_len_rel', 0.10, ...
    'smooth_len_rel', 0.05, ...
    'ms_weight', [0.3, 0.7], ...
    'anisotropy', [3.0, 1.0], ...
    'coupling', 0.5, ...
    'noise_level', 0.0, ...
    'noise_granularity', 0.5, ...
    'noise_bias', 0.5, ...
    'k_mean', 5e-9, ...
    'var_rel', 0.5, ...
    'a_max', 2.0, ...
    'a_gamma', 2.0, ...
    'tensor_strength', 1.0, ...
    'theta_jitter', 0.01, ...
    'theta_smooth_rel', 0.10, ...
    'eps_smooth_rel', 0.05, ...
    'texture_amp', 0.005, ...
    'p_inlet_mean', 350, ...
    'a_sin', 0.03, ...
    'f_sin', 0.75, ...
    'phi_sin', pi, ...
    'k_gauss', 2, ...
    'a_gauss', 0.05, ...
    'sigma_gauss', 0.05, ...
    'gauss_jitter', 0.25, ...
    'a_lin', 0.025, ...
    'save', true, ...
    'delimiter', ';', ...
    'save_dir', output_dir, ...
    'file_tag', file_tag, ...
    'packing_batch_variation', 0.8, ...
    'packing_batch_seed', 3001, ...
    'packing_case_id', 1);
end

function z = expected_packing_scatter(seed)
stream = RandStream('mt19937ar', 'Seed', seed);
u = rand(stream, 1, 1);
probability = eps + (1 - 2 * eps) * u;
z = truncated_standard_normal_quantile(probability);
end

function z = truncated_standard_normal_quantile(probability)
lower = -3;
upper = 3;
phi = @(value) 0.5 * (1 + erf(value / sqrt(2)));
truncated_probability = phi(lower) + ...
    (phi(upper) - phi(lower)) * probability;
z = sqrt(2) * erfinv(2 * truncated_probability - 1);
end

function cleanup_temp_directory(output_dir)
if isfolder(output_dir)
    [ok, message] = rmdir(output_dir, 's');
    assert(ok, 'test_gen_simulation_inputs_smoke:CleanupFailed', ...
        'Could not remove temporary directory %s: %s', output_dir, message);
end
end
