function tests = test_gen_simulation_inputs_smoke
% Generator-only contract smoke test; does not start COMSOL.
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
verifyGreaterThanOrEqual(testCase, min(porosity(:)), opts.eps_min_global);
verifyLessThanOrEqual(testCase, max(porosity(:)), opts.eps_max_global);
verifyGreaterThan(testCase, min(porosity(:)), eps);
verifyFalse(testCase, any(porosity(:) == eps));
verifyEqual(testCase, info_a.porosity.parameters.eps_min_global, opts.eps_min_global);
verifyEqual(testCase, info_a.porosity.parameters.eps_max_global, opts.eps_max_global);

porosity_ref = info_a.porosity.parameters.eps_ref;
kc_at_reference = (opts.A_rel * opts.k_mean) * porosity_ref^3 / ...
    max((1 - porosity_ref)^2, eps);
verifyEqual(testCase, kc_at_reference, opts.k_mean, 'RelTol', 1e-12);

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
    'A_rel', 2.0, ...
    'eps_min_global', 0.30, ...
    'eps_max_global', 0.80, ...
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
    'file_tag', file_tag);
end

function cleanup_temp_directory(output_dir)
if isfolder(output_dir)
    [ok, message] = rmdir(output_dir, 's');
    assert(ok, 'test_gen_simulation_inputs_smoke:CleanupFailed', ...
        'Could not remove temporary directory %s: %s', output_dir, message);
end
end
