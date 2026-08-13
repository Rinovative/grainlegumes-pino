%% test_sample_parameters_contract.m
% ============================================================
% Verify the current sampling schema and deterministic persistence.
%
% Author: Rino M. Albertin
% Date:   2026-08-13
%
% DESCRIPTION
%   Exercises all supported sampling methods against the exact ordered design,
%   version-1 generation contract, deterministic CSV values, and JSON identity.
%
% REQUIREMENTS
%   MATLAB Unit Test framework and sampling toolboxes. COMSOL is not required.
% ============================================================

function tests = test_sample_parameters_contract

tests = functiontests(localfunctions);
end

function testCurrentDoeSchemaAndDeterminism(testCase)
this_dir = fileparts(mfilename('fullpath'));
core_dir = fullfile(fileparts(this_dir), 'core');
original_path = path;
path_cleanup = onCleanup(@() path(original_path)); %#ok<NASGU>
addpath(genpath(core_dir));

output_dir = tempname;
mkdir(output_dir);
output_cleanup = onCleanup(@() cleanup_temp_directory(output_dir)); %#ok<NASGU>

[base, param_names, contract] = sample_parameters('contract', 0.8);
for method = ["uniform", "lhs", "sobol"]
    sample_parameters(char(method), 0.8, 4, 3001, output_dir);
    filename = sprintf('%s_var80_seed3001', method);
    path_csv = fullfile(output_dir, filename + ".csv");
    path_json = fullfile(output_dir, filename + ".json");
    first = readmatrix(path_csv, 'Delimiter', ';');
    payload = jsondecode(fileread(path_json));
    T = readtable(path_csv, 'Delimiter', ';');
    expected_columns = ["case_id", param_names];
    if method == "sobol"
        expected_columns(end + 1) = "simulate";
    end
    verifyEqual(testCase, string(T.Properties.VariableNames), expected_columns);
    verifyEqual(testCase, string(payload.meta.param_names(:)), param_names(:));
    verifyEqual(testCase, payload.meta.base, base);
    verifyEqual(testCase, payload.meta.generation_contract, contract);
    verifyGreaterThanOrEqual(testCase, min(T.k_mean), contract.batch_kappa_support(1));
    verifyLessThanOrEqual(testCase, max(T.k_mean), contract.batch_kappa_support(2));
    sample_parameters(char(method), 0.8, 4, 3001, output_dir);
    verifyEqual(testCase, readmatrix(path_csv, 'Delimiter', ';'), first);
end
end

function cleanup_temp_directory(output_dir)
if isfolder(output_dir)
    [ok, message] = rmdir(output_dir, 's');
    assert(ok, 'test_sample_parameters_contract:CleanupFailed', ...
        'Could not remove temporary directory %s: %s', output_dir, message);
end
end
