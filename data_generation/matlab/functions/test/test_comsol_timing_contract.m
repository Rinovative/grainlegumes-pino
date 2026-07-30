function tests = test_comsol_timing_contract
% COMSOL-free tests for solve timing validation and atomic persistence.
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
this_dir = fileparts(mfilename('fullpath'));
core_dir = fullfile(fileparts(this_dir), 'core');
testCase.TestData.original_path = path;
addpath(core_dir);
end

function teardownOnce(testCase)
path(testCase.TestData.original_path);
end

function testCasesAreSortedAndAggregatesAreDerived(testCase)
cases = [timing_case("case_0002", 4); timing_case("case_0001", 2)];
payload = comsol_solve_timing( ...
    "batch_a", cases, runtime_fixture(), repmat('a', 1, 64));
verifyEqual(testCase, string({payload.cases.case_id})', ...
    ["case_0001"; "case_0002"]);
verifyEqual(testCase, payload.aggregates.measured_case_count, 2);
verifyEqual(testCase, payload.aggregates.mean_s, 3);
verifyEqual(testCase, payload.aggregates.median_s, 3);
verifyEqual(testCase, payload.aggregates.p10_s, 2.2, 'AbsTol', 1e-12);
verifyEqual(testCase, payload.aggregates.p90_s, 3.8, 'AbsTol', 1e-12);
end

function testInvalidCasesAreRejected(testCase)
invalid = timing_case("case_0001", 1);
invalid.comsol_solve_s = 0;
verifyError(testCase, @() comsol_solve_timing( ...
    "batch_a", invalid, runtime_fixture(), ""), ...
    'comsol_solve_timing:Duration');
invalid.comsol_solve_s = -1;
verifyError(testCase, @() comsol_solve_timing( ...
    "batch_a", invalid, runtime_fixture(), ""), ...
    'comsol_solve_timing:Duration');
invalid.comsol_solve_s = Inf;
verifyError(testCase, @() comsol_solve_timing( ...
    "batch_a", invalid, runtime_fixture(), ""), ...
    'comsol_solve_timing:Duration');
malformed = rmfield(timing_case("case_0001", 1), 'comsol_solve_s');
verifyError(testCase, @() comsol_solve_timing( ...
    "batch_a", malformed, runtime_fixture(), ""), ...
    'comsol_solve_timing:CaseSchema');
duplicate = [timing_case("case_0001", 1); ...
    timing_case("case_0001", 2)];
verifyError(testCase, @() comsol_solve_timing( ...
    "batch_a", duplicate, runtime_fixture(), ""), ...
    'comsol_solve_timing:DuplicateCase');
end

function testAtomicJsonRoundTripUsesProcessedBatchDirectory(testCase)
generated_data_root = tempname;
processed_dir = fullfile(generated_data_root, 'processed', 'batch_a');
mkdir(processed_dir);
cleanup = onCleanup(@() rmdir(generated_data_root, 's')); %#ok<NASGU>
path_json = fullfile(processed_dir, 'comsol_solve_timing.json');
payload = comsol_solve_timing( ...
    "batch_a", timing_case("case_0001", 2), runtime_fixture(), ...
    repmat('b', 1, 64), path_json);
verifyTrue(testCase, isfile(path_json));
verifyFalse(testCase, isfile([path_json '.tmp']));
decoded = jsondecode(fileread(path_json));
verifyEqual(testCase, decoded.schema_kind, 'comsol_solve_timing');
verifyEqual(testCase, decoded.schema_version, 1);
verifyEqual(testCase, decoded.cases.comsol_solve_s, 2);
verifyEqual(testCase, decoded.aggregates.mean_s, payload.aggregates.mean_s);
verifyNotEmpty(testCase, regexp(fileread(path_json), ...
    '"cases"\s*:\s*\[', 'once'));
end

function record = timing_case(case_id, duration)
record = struct('case_id', case_id, 'comsol_solve_s', duration);
end

function runtime = runtime_fixture()
runtime = struct( ...
    'matlab_version', "test", ...
    'comsol_version', "test", ...
    'os', "test", ...
    'hostname', "test-host", ...
    'processor', "test-cpu", ...
    'case_execution', "sequential");
end
