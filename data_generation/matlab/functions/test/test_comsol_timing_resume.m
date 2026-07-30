function tests = test_comsol_timing_resume
% COMSOL-free tests for timing resume, private progress, and repair.
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
this_dir = fileparts(mfilename('fullpath'));
core_dir = fullfile(fileparts(this_dir), 'core');
matlab_dir = fileparts(fileparts(this_dir));
testCase.TestData.original_path = path;
testCase.TestData.core_dir = core_dir;
testCase.TestData.batch_run_path = fullfile(matlab_dir, 'batch_run.m');
addpath(core_dir);
end

function teardownOnce(testCase)
path(testCase.TestData.original_path);
end

function testBatchProducerLockIsExclusiveAndReleasable(testCase)
root = tempname;
mkdir(root);
cleanup = onCleanup(@() rmdir(root, 's')); %#ok<NASGU>
lock_path = fullfile(root, 'batch.lock');
first_lock_cleanup = acquire_comsol_batch_lock(lock_path);
verifyError(testCase, @() acquire_comsol_batch_lock(lock_path), ...
    'acquire_comsol_batch_lock:Unavailable');
clear first_lock_cleanup
second_lock_cleanup = acquire_comsol_batch_lock(lock_path); %#ok<NASGU>
verifyTrue(testCase, isfile(lock_path));
clear second_lock_cleanup
end

function testTimingResumeAcceptsEquivalentTextRepresentations(testCase)
prior = runtime_fixture();
fields = fieldnames(prior);
for field_index = 1:numel(fields)
    prior.(fields{field_index}) = char(prior.(fields{field_index}));
end
current = runtime_fixture();
partial = timing_case("case_0001", 1);

[cases, selected_runtime] = prepare_comsol_timing_resume( ...
    partial, prior, current, "case_0001", ...
    ["case_0001"; "case_0002"]);

verifyEqual(testCase, cases.case_id, "case_0001");
verifyEqual(testCase, selected_runtime, prior);
end

function testTimingResumeRejectsMalformedRuntime(testCase)
malformed = rmfield(runtime_fixture(), 'processor');
partial = timing_case("case_0001", 1);
verifyError(testCase, @() prepare_comsol_timing_resume( ...
    partial, malformed, runtime_fixture(), "case_0001", ...
    ["case_0001"; "case_0002"]), ...
    'prepare_comsol_timing_resume:RuntimeSchema');
end

function testTimingResumeUsesCurrentRuntimeForEmptySidecar(testCase)
prior_runtime = runtime_fixture();
prior_runtime.hostname = "old-host";
current_runtime = runtime_fixture();
empty_cases = repmat(timing_case("case_0001", 1), 0, 1);

[cases, selected_runtime] = prepare_comsol_timing_resume( ...
    empty_cases, prior_runtime, current_runtime, strings(0, 1), ...
    ["case_0001"; "case_0002"]);

verifyEmpty(testCase, cases);
verifyEqual(testCase, selected_runtime, current_runtime);
end

function testTimingResumeDropsStaleOnlyTiming(testCase)
prior_runtime = runtime_fixture();
prior_runtime.hostname = "old-host";
current_runtime = runtime_fixture();
stale = timing_case("case_0001", 1);

[cases, selected_runtime] = prepare_comsol_timing_resume( ...
    stale, prior_runtime, current_runtime, "case_0002", ...
    ["case_0001"; "case_0002"]);

verifyEmpty(testCase, cases);
verifyEqual(testCase, selected_runtime, current_runtime);
end

function testTimingResumeRejectsPartialReusableRuntimeMismatch(testCase)
prior_runtime = runtime_fixture();
prior_runtime.hostname = "old-host";
current_runtime = runtime_fixture();
partial = timing_case("case_0001", 1);

verifyError(testCase, @() prepare_comsol_timing_resume( ...
    partial, prior_runtime, current_runtime, "case_0001", ...
    ["case_0001"; "case_0002"]), ...
    'batch_run:TimingRuntimeMismatch');
end

function testTimingResumeRetainsFullCoverageWithoutCurrentMatch(testCase)
prior_runtime = runtime_fixture();
prior_runtime.hostname = "old-host";
current_runtime = runtime_fixture();
full = [timing_case("case_0002", 2); timing_case("case_0001", 1)];

[cases, selected_runtime] = prepare_comsol_timing_resume( ...
    full, prior_runtime, current_runtime, ...
    ["case_0001"; "case_0002"], ["case_0001"; "case_0002"]);

verifyEqual(testCase, string({cases.case_id})', ...
    ["case_0002"; "case_0001"]);
verifyEqual(testCase, selected_runtime, prior_runtime);
end

function testInterruptedTimingCheckpointPreservesManifestOrder(testCase)
root = tempname;
mkdir(root);
cleanup = onCleanup(@() rmdir(root, 's')); %#ok<NASGU>
timing_path = fullfile(root, 'comsol_solve_timing.json');
intended_case_ids = ["case_0003"; "case_0001"; "case_0002"];
existing = timing_case("case_0002", 7.25);

partial = comsol_solve_timing( ...
    "batch_a", existing, runtime_fixture(), "", timing_path, ...
    intended_case_ids);
verifyEqual(testCase, partial.aggregates.measured_case_count, 1);
verifyEqual(testCase, partial.cases.comsol_solve_s, 7.25);

continued = [partial.cases; timing_case("case_0003", 4.5)];
continued_payload = comsol_solve_timing( ...
    "batch_a", continued, runtime_fixture(), "", timing_path, ...
    intended_case_ids);
verifyEqual(testCase, string({continued_payload.cases.case_id})', ...
    ["case_0003"; "case_0002"]);
retained = continued_payload.cases( ...
    string({continued_payload.cases.case_id}) == "case_0002");
verifyEqual(testCase, retained.comsol_solve_s, 7.25);
verifyEqual(testCase, ...
    continued_payload.aggregates.measured_case_count, 2);
verifyFalse(testCase, isfile([timing_path '.tmp']));

decoded = jsondecode(fileread(timing_path));
verifyEqual(testCase, string({decoded.cases.case_id})', ...
    ["case_0003"; "case_0002"]);
verifyEmpty(testCase, decoded.batch_manifest_sha256);
end

function testPrivateProgressCheckpointsReusableCasesAndTiming(testCase)
root = tempname;
mkdir(root);
cleanup = onCleanup(@() rmdir(root, 's')); %#ok<NASGU>
progress_path = fullfile(root, 'batch_progress.json');
manifest_path = fullfile(root, 'batch_manifest.json');
timing_path = fullfile(root, 'comsol_solve_timing.json');
case_ids = ["case_0001"; "case_0002"];
records = [pending_record("case_0001"); pending_record("case_0002")];
configuration = struct('save_model', false);
field_schema = struct('input_columns', {{'x'}}, ...
    'solution_columns', {{'x'; 'p'}});

[progress, progress_timing] = publish_comsol_batch_progress( ...
    progress_path, timing_path, "batch_a", configuration, field_schema, ...
    case_ids, records, repmat(timing_case("case_0001", 1), 0, 1), ...
    runtime_fixture());
verifyEqual(testCase, progress.schema_kind, "comsol_batch_progress");
verifyEqual(testCase, progress.schema_version, 1);
verifyEqual(testCase, progress.status, "in_progress");
verifyEmpty(testCase, progress_timing.cases);
verifyEqual(testCase, progress_timing.batch_manifest_sha256, "");
verifyFalse(testCase, isfile(manifest_path));

records(1) = complete_record("case_0001", repmat('a', 1, 64));
[progress, progress_timing] = publish_comsol_batch_progress( ...
    progress_path, timing_path, "batch_a", configuration, field_schema, ...
    case_ids, records, timing_case("case_0001", 2.5), ...
    runtime_fixture());
decoded_progress = jsondecode(fileread(progress_path));
decoded_timing = jsondecode(fileread(timing_path));
verifyEqual(testCase, decoded_progress.schema_kind, ...
    'comsol_batch_progress');
verifyEqual(testCase, decoded_progress.schema_version, 1);
verifyEqual(testCase, string({decoded_progress.cases.status})', ...
    ["complete"; "pending"]);
verifyEqual(testCase, progress.cases(1).files.raw_csv_sha256, ...
    repmat('a', 1, 64));
verifyEqual(testCase, progress_timing.cases.case_id, "case_0001");
verifyEmpty(testCase, decoded_timing.batch_manifest_sha256);
verifyFalse(testCase, isfile([progress_path '.tmp']));
verifyFalse(testCase, isfile([timing_path '.tmp']));
verifyFalse(testCase, isfile(manifest_path));
end

function testPrivateProgressRejectsPendingCaseDigests(testCase)
root = tempname;
mkdir(root);
cleanup = onCleanup(@() rmdir(root, 's')); %#ok<NASGU>
progress_path = fullfile(root, 'batch_progress.json');
timing_path = fullfile(root, 'comsol_solve_timing.json');
records = pending_record("case_0001");
records.files.raw_csv_sha256 = repmat('a', 1, 64);
configuration = struct('save_model', false);
field_schema = struct('input_columns', {{'x'}}, ...
    'solution_columns', {{'x'; 'p'}});
arguments = { ...
    progress_path, timing_path, "batch_a", configuration, field_schema, ...
    "case_0001", records, ...
    repmat(timing_case("case_0001", 1), 0, 1), runtime_fixture()};

verifyError(testCase, @() publish_comsol_batch_progress(arguments{:}), ...
    'publish_comsol_batch_progress:CaseDigest');
verifyFalse(testCase, isfile(progress_path));
verifyFalse(testCase, isfile(timing_path));
end

function testFinalFullCoverageUsesManifestOrder(testCase)
intended_case_ids = ["case_0003"; "case_0001"; "case_0002"];
cases = [ ...
    timing_case("case_0002", 2); ...
    timing_case("case_0003", 3); ...
    timing_case("case_0001", 1)];
payload = comsol_solve_timing( ...
    "batch_a", cases, runtime_fixture(), repmat('a', 1, 64), "", ...
    intended_case_ids);
verifyEqual(testCase, string({payload.cases.case_id})', ...
    intended_case_ids);
verifyEqual(testCase, payload.aggregates.measured_case_count, 3);
end

function testTimingMembershipAndDuplicatesAreRejected(testCase)
intended_case_ids = ["case_0001"; "case_0002"];
verifyError(testCase, @() comsol_solve_timing( ...
    "batch_a", timing_case("case_0003", 1), runtime_fixture(), ...
    "", "", intended_case_ids), ...
    'comsol_solve_timing:CaseMembership');
duplicates = [ ...
    timing_case("case_0001", 1); ...
    timing_case("case_0001", 2)];
verifyError(testCase, @() comsol_solve_timing( ...
    "batch_a", duplicates, runtime_fixture(), "", "", ...
    intended_case_ids), ...
    'comsol_solve_timing:DuplicateCase');
end

function testPrivateRepairPreservesRawAndCanonicalOutputs(testCase)
fixture = repair_fixture();
cleanup = onCleanup(@() rmdir(fixture.root, 's')); %#ok<NASGU>
raw_before = fileread(fixture.field_path);
metadata_before = fileread(fixture.metadata_path);
solution_before = fileread(fixture.solution_path);
stale_private_dir = fullfile( ...
    fixture.processed_dir, '.case_0001_timing_repair');
mkdir(stale_private_dir);
write_text(fullfile(stale_private_dir, 'interrupted.tmp'), ...
    'stale-private-output');

results = repair_comsol_case_timing( ...
    fixture.field_path, fixture.template_path, fixture.processed_dir, ...
    false, @successful_private_runner);

verifyEqual(testCase, results.comsol_solve_s, 12.5);
verifyEqual(testCase, fileread(fixture.field_path), raw_before);
verifyEqual(testCase, fileread(fixture.metadata_path), metadata_before);
verifyEqual(testCase, fileread(fixture.solution_path), solution_before);
verifyEqual(testCase, directory_names(fixture.processed_dir), ...
    "case_0001_sol.csv");
end

function testFailedPrivateRepairLeavesCanonicalOutputIntact(testCase)
fixture = repair_fixture();
cleanup = onCleanup(@() rmdir(fixture.root, 's')); %#ok<NASGU>
raw_before = fileread(fixture.field_path);
solution_before = fileread(fixture.solution_path);

verifyError(testCase, @() repair_comsol_case_timing( ...
    fixture.field_path, fixture.template_path, fixture.processed_dir, ...
    false, @failing_private_runner), 'test:PrivateSolve');

verifyEqual(testCase, fileread(fixture.field_path), raw_before);
verifyEqual(testCase, fileread(fixture.solution_path), solution_before);
verifyEqual(testCase, directory_names(fixture.processed_dir), ...
    "case_0001_sol.csv");
end

function testPostSolveValidationFailureStillDetectsCanonicalMutation(testCase)
fixture = repair_fixture();
cleanup = onCleanup(@() rmdir(fixture.root, 's')); %#ok<NASGU>

verifyError(testCase, @() repair_comsol_case_timing( ...
    fixture.field_path, fixture.template_path, fixture.processed_dir, ...
    false, @mutating_invalid_runner), ...
    'repair_comsol_case_timing:CanonicalMutation');
end

function testThrownSolveFailureStillDetectsCanonicalMutation(testCase)
fixture = repair_fixture();
cleanup = onCleanup(@() rmdir(fixture.root, 's')); %#ok<NASGU>

verifyError(testCase, @() repair_comsol_case_timing( ...
    fixture.field_path, fixture.template_path, fixture.processed_dir, ...
    false, @mutating_failing_runner), ...
    'repair_comsol_case_timing:CanonicalMutation');
end

function testAuthoritativeSolveTimerBoundaryIsUnchanged(testCase)
source = fileread(fullfile( ...
    testCase.TestData.core_dir, 'run_comsol_case.m'));
boundary = regexp(source, ...
    ['solve_timer\s*=\s*tic;\s*' ...
    'mphrun\(model\);\s*' ...
    'comsol_solve_s\s*=\s*toc\(solve_timer\);'], ...
    'once');
verifyNotEmpty(testCase, boundary);
end

function fixture = repair_fixture()
root = tempname;
raw_dir = fullfile(root, 'raw');
processed_dir = fullfile(root, 'processed');
mkdir(raw_dir);
mkdir(processed_dir);
field_path = fullfile(raw_dir, 'case_0001.csv');
metadata_path = fullfile(raw_dir, 'case_0001.json');
template_path = fullfile(root, 'template.mph');
solution_path = fullfile(processed_dir, 'case_0001_sol.csv');
write_text(field_path, 'raw-field-bytes');
write_text(metadata_path, '{"case_id":"case_0001"}');
write_text(template_path, 'template-bytes');
write_text(solution_path, 'canonical-solution-bytes');
fixture = struct( ...
    'root', root, ...
    'field_path', field_path, ...
    'metadata_path', metadata_path, ...
    'template_path', template_path, ...
    'processed_dir', processed_dir, ...
    'solution_path', solution_path);
end

function results = successful_private_runner( ...
        field_path, template_path, output_dir, save_model) %#ok<INUSD>
[~, case_id] = fileparts(field_path);
write_text(fullfile(output_dir, [case_id '_sol.csv']), ...
    'private-solution-bytes');
if save_model
    write_text(fullfile(output_dir, [case_id '_sol.mph']), ...
        'private-model-bytes');
end
results = struct('comsol_solve_s', 12.5, 'time_s', 13.0);
end

function results = mutating_invalid_runner( ...
        field_path, template_path, output_dir, save_model) %#ok<INUSD>
[~, case_id] = fileparts(field_path);
canonical_path = fullfile(fileparts(output_dir), [case_id '_sol.csv']);
write_text(canonical_path, 'mutated-canonical-output');
write_text(fullfile(output_dir, [case_id '_sol.csv']), ...
    'private-solution-bytes');
results = struct('comsol_solve_s', 0);
end

function results = mutating_failing_runner( ...
        field_path, template_path, output_dir, save_model) %#ok<INUSD,STOUT>
[~, case_id] = fileparts(field_path);
canonical_path = fullfile(fileparts(output_dir), [case_id '_sol.csv']);
write_text(canonical_path, 'mutated-canonical-output');
error('test:PrivateSolve', 'Synthetic private solve failure.');
end

function results = failing_private_runner( ...
        field_path, template_path, output_dir, save_model) %#ok<INUSD,STOUT>
[~, case_id] = fileparts(field_path);
write_text(fullfile(output_dir, [case_id '_sol.csv']), ...
    'incomplete-private-output');
error('test:PrivateSolve', 'Synthetic private solve failure.');
end

function names = directory_names(path)
entries = dir(path);
is_real_entry = ~ismember(string({entries.name}), ["."; ".."])';
entries = entries(is_real_entry);
names = sort(string({entries.name}))';
end

function write_text(path, text)
fid = fopen(path, 'w');
if fid < 0
    error('test:FixtureWrite', 'Could not write fixture: %s', path);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s', text);
end

function record = pending_record(case_id)
empty_hashes = struct( ...
    'raw_csv_sha256', "", ...
    'raw_json_sha256', "", ...
    'solution_csv_sha256', "", ...
    'solution_model_sha256', "");
record = struct( ...
    'case_id', case_id, ...
    'status', "pending", ...
    'stage', "", ...
    'message', "", ...
    'files', empty_hashes);
end

function record = complete_record(case_id, digest)
files = struct( ...
    'raw_csv_sha256', string(digest), ...
    'raw_json_sha256', string(digest), ...
    'solution_csv_sha256', string(digest), ...
    'solution_model_sha256', "");
record = struct( ...
    'case_id', case_id, ...
    'status', "complete", ...
    'stage', "simulation", ...
    'message', "", ...
    'files', files);
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
