function [reusable_timings, timing_runtime] = ...
        prepare_comsol_timing_resume( ...
        prior_timings, prior_runtime, current_runtime, ...
        prior_complete_case_ids, intended_case_ids)
%PREPARE_COMSOL_TIMING_RESUME Retain only scientifically reusable timings.
% A single timing sidecar describes one runtime. Existing measurements may be
% extended only on that exact runtime; stale or empty measurements never lend
% their provenance to newly measured cases.

prior_complete_case_ids = string(prior_complete_case_ids(:));
intended_case_ids = string(intended_case_ids(:));
if isempty(prior_timings)
    reusable_timings = prior_timings;
else
    prior_case_ids = string({prior_timings.case_id})';
    reusable_timings = prior_timings( ...
        ismember(prior_case_ids, prior_complete_case_ids));
    reusable_timings = reusable_timings(:);
end

if isempty(reusable_timings)
    timing_runtime = current_runtime;
    return;
end

reusable_case_ids = string({reusable_timings.case_id})';
full_coverage = numel(reusable_case_ids) == numel(intended_case_ids) && ...
    all(ismember(intended_case_ids, reusable_case_ids));
if isempty(prior_runtime)
    timing_runtime_mismatch();
end
if ~full_coverage && ~timing_runtime_equal(prior_runtime, current_runtime)
    timing_runtime_mismatch();
end
timing_runtime = prior_runtime;
end

function equal = timing_runtime_equal(left, right)
fields = { ...
    'matlab_version'; 'comsol_version'; 'os'; 'hostname'; ...
    'processor'; 'case_execution'};
validate_runtime(left, fields, 'prior');
validate_runtime(right, fields, 'current');
equal = true;
for field_index = 1:numel(fields)
    field = fields{field_index};
    equal = equal && strcmp( ...
        char(string(left.(field))), char(string(right.(field))));
end
end

function validate_runtime(runtime, fields, label)
if ~isstruct(runtime) || ~isscalar(runtime) || ...
        ~isequal(sort(fieldnames(runtime)), sort(fields))
    error('prepare_comsol_timing_resume:RuntimeSchema', ...
        '%s runtime must contain the exact timing provenance fields.', label);
end
for field_index = 1:numel(fields)
    value = runtime.(fields{field_index});
    if ~((ischar(value) && isrow(value) && ~isempty(value)) || ...
            (isstring(value) && isscalar(value) && ~ismissing(value) && ...
            strlength(value) > 0))
        error('prepare_comsol_timing_resume:RuntimeValue', ...
            '%s runtime fields must be non-empty text scalars.', label);
    end
end
if ~strcmp(char(string(runtime.case_execution)), 'sequential')
    error('prepare_comsol_timing_resume:RuntimeValue', ...
        '%s runtime case_execution must be sequential.', label);
end
end

function timing_runtime_mismatch()
error('batch_run:TimingRuntimeMismatch', ...
    ['Existing reusable timings use a different MATLAB/COMSOL/host runtime. ' ...
    'The single-runtime sidecar cannot combine those measurements; ' ...
    'resume timing on the original runtime.']);
end
