function payload = comsol_solve_timing( ...
        batch_name, cases, runtime, batch_manifest_sha256, output_path)
%COMSOL_SOLVE_TIMING Build, validate, and optionally publish solve timings.
% Per-case timings are operational metadata and never change case identity.
% batch_run publishes this sidecar beside the processed COMSOL results.

if nargin < 5
    output_path = "";
end
batch_name = require_text(batch_name, 'batch_name', false);
manifest_digest = require_text( ...
    batch_manifest_sha256, 'batch_manifest_sha256', true);
if ~isempty(manifest_digest) && ...
        isempty(regexp(manifest_digest, '^[0-9a-f]{64}$', 'once'))
    error('comsol_solve_timing:ManifestDigest', ...
        'batch_manifest_sha256 must be empty or a lowercase SHA-256.');
end
validated_runtime = validate_runtime(runtime);
validated_cases = validate_cases(cases);
durations = [validated_cases.comsol_solve_s]';
aggregates = summarize(durations);
payload = struct( ...
    'schema_kind', "comsol_solve_timing", ...
    'schema_version', 1, ...
    'batch_name', string(batch_name), ...
    'batch_manifest_sha256', string(manifest_digest), ...
    'runtime', validated_runtime, ...
    'cases', validated_cases, ...
    'aggregates', aggregates);

path = require_text(output_path, 'output_path', true);
if isempty(path)
    return;
end
serializable = payload;
serializable.cases = reshape(num2cell(validated_cases), [], 1);
write_json_atomic(path, serializable);
end

function cases = validate_cases(cases)
required = {'case_id'; 'comsol_solve_s'};
if isempty(cases)
    cases = repmat(struct('case_id', "", 'comsol_solve_s', []), 0, 1);
    return;
end
if ~isstruct(cases)
    error('comsol_solve_timing:Cases', ...
        'cases must be a structure array.');
end
case_ids = strings(numel(cases), 1);
for case_index = 1:numel(cases)
    record = cases(case_index);
    if ~isequal(sort(fieldnames(record)), sort(required))
        error('comsol_solve_timing:CaseSchema', ...
            'Each timing case must contain case_id and comsol_solve_s.');
    end
    case_ids(case_index) = string(require_text( ...
        record.case_id, 'case_id', false));
    if isempty(regexp(char(case_ids(case_index)), ...
            '^case_[0-9]{4}$', 'once'))
        error('comsol_solve_timing:CaseId', ...
            'case_id must use the canonical case_0000 form.');
    end
    if ~is_positive_duration(record.comsol_solve_s)
        error('comsol_solve_timing:Duration', ...
            'comsol_solve_s must be a finite positive real scalar.');
    end
end
if numel(unique(case_ids)) ~= numel(case_ids)
    error('comsol_solve_timing:DuplicateCase', ...
        'COMSOL solve timing case IDs must be unique.');
end
[~, order] = sort(case_ids);
cases = cases(order);
cases = cases(:);
end

function runtime = validate_runtime(runtime)
required = {'matlab_version'; 'comsol_version'; 'os'; 'hostname'; ...
    'processor'; 'case_execution'};
if ~isstruct(runtime) || ~isscalar(runtime) || ...
        ~isequal(sort(fieldnames(runtime)), sort(required))
    error('comsol_solve_timing:Runtime', ...
        'runtime must contain the exact execution-provenance fields.');
end
for field_index = 1:numel(required)
    field = required{field_index};
    require_text(runtime.(field), ['runtime.' field], false);
end
if ~strcmp(char(string(runtime.case_execution)), 'sequential')
    error('comsol_solve_timing:Runtime', ...
        'runtime.case_execution must be sequential.');
end
end

function summary = summarize(values)
summary = struct( ...
    'measured_case_count', numel(values), ...
    'mean_s', optional_mean(values), ...
    'median_s', optional_percentile(values, 50), ...
    'p10_s', optional_percentile(values, 10), ...
    'p90_s', optional_percentile(values, 90));
end

function value = optional_mean(values)
if isempty(values), value = []; else, value = mean(values); end
end

function value = optional_percentile(values, percentile)
if isempty(values)
    value = [];
    return;
end
sorted_values = sort(values(:));
if numel(sorted_values) == 1
    value = sorted_values(1);
    return;
end
position = 1 + (numel(sorted_values) - 1) * percentile / 100;
lower_index = floor(position);
upper_index = ceil(position);
weight = position - lower_index;
value = sorted_values(lower_index) * (1 - weight) + ...
    sorted_values(upper_index) * weight;
end

function tf = is_positive_duration(value)
tf = isnumeric(value) && ~islogical(value) && isreal(value) && ...
    isscalar(value) && isfinite(value) && value > 0;
end

function text = require_text(value, label, allow_empty)
if ischar(value) && (isrow(value) || isempty(value))
    text = value;
elseif isstring(value) && isscalar(value) && ~ismissing(value)
    text = char(value);
else
    error('comsol_solve_timing:Text', ...
        '%s must be a text scalar.', label);
end
if ~allow_empty && isempty(text)
    error('comsol_solve_timing:Text', ...
        '%s must be non-empty.', label);
end
end

function write_json_atomic(path, payload)
temp_path = [path '.tmp'];
if isfile(temp_path), delete(temp_path); end
fid = fopen(temp_path, 'w');
if fid < 0
    error('comsol_solve_timing:Open', ...
        'Could not open temporary timing sidecar: %s', temp_path);
end
try
    fprintf(fid, '%s', jsonencode(payload, 'PrettyPrint', true));
    fclose(fid);
catch write_error
    fclose(fid);
    if isfile(temp_path), delete(temp_path); end
    rethrow(write_error);
end
[move_ok, move_message] = movefile(temp_path, path);
if ~move_ok
    if isfile(temp_path), delete(temp_path); end
    error('comsol_solve_timing:Publish', ...
        'Failed to publish timing sidecar: %s', move_message);
end
end
