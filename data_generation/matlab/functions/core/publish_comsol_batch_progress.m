function [progress, timing_payload] = publish_comsol_batch_progress( ...
        progress_path, timing_path, batch_name, configuration, ...
        field_schema, intended_case_ids, case_records, solve_timings, runtime)
%PUBLISH_COMSOL_BATCH_PROGRESS Atomically checkpoint private resume state.
% The terminal batch manifest is never used for in-progress state. This
% version-1 progress file fences consumers from an active or interrupted batch.
% Timing remains a separate version-1 sidecar with an empty terminal-manifest
% digest until final manifest publication.

progress_path = require_text(progress_path, 'progress_path', false);
timing_path = require_text(timing_path, 'timing_path', false);
batch_name = require_text(batch_name, 'batch_name', false);
if ~isstruct(configuration) || ~isscalar(configuration) || ...
        ~isfield(configuration, 'save_model') || ...
        ~islogical(configuration.save_model) || ...
        ~isscalar(configuration.save_model)
    error('publish_comsol_batch_progress:Configuration', ...
        'configuration must be a scalar struct with logical save_model.');
end
if ~isstruct(field_schema) || ~isscalar(field_schema)
    error('publish_comsol_batch_progress:FieldSchema', ...
        'field_schema must be a scalar struct.');
end
case_ids = validate_case_ids(intended_case_ids);
validated_records = validate_records( ...
    case_records, case_ids, configuration.save_model);

progress = struct( ...
    'schema_kind', "comsol_batch_progress", ...
    'schema_version', 1, ...
    'batch_name', string(batch_name), ...
    'status', "in_progress", ...
    'configuration', configuration, ...
    'field_schema', field_schema, ...
    'intended_case_ids', {cellstr(case_ids)}, ...
    'cases', validated_records);

serializable = progress;
serializable.cases = reshape(num2cell(validated_records), [], 1);
write_json_atomic(progress_path, serializable);
timing_payload = comsol_solve_timing( ...
    batch_name, solve_timings, runtime, "", timing_path, case_ids);
end

function records = validate_records(records, case_ids, save_model)
required_record_fields = {'case_id'; 'status'; 'stage'; 'message'; 'files'};
required_file_fields = { ...
    'raw_csv_sha256'; 'raw_json_sha256'; ...
    'solution_csv_sha256'; 'solution_model_sha256'};
if ~isstruct(records) || numel(records) ~= numel(case_ids)
    error('publish_comsol_batch_progress:Cases', ...
        'case_records must contain one ordered record per intended case.');
end
records = records(:);
for record_index = 1:numel(records)
    record = records(record_index);
    if ~isequal(sort(fieldnames(record)), sort(required_record_fields))
        error('publish_comsol_batch_progress:CaseSchema', ...
            'Every case record must contain the exact required fields.');
    end
    case_id = require_text(record.case_id, 'case record case_id', false);
    if ~strcmp(case_id, char(case_ids(record_index)))
        error('publish_comsol_batch_progress:CaseOrder', ...
            'case_records must follow intended_case_ids exactly.');
    end
    record_status = require_text( ...
        record.status, 'case record status', false);
    stage = require_text(record.stage, 'case record stage', true);
    message = require_text(record.message, 'case record message', true);
    if ~isstruct(record.files) || ~isscalar(record.files) || ...
            ~isequal(sort(fieldnames(record.files)), ...
            sort(required_file_fields))
        error('publish_comsol_batch_progress:FileSchema', ...
            'Every case files record must contain the exact hash fields.');
    end
    hashes = { ...
        record.files.raw_csv_sha256, ...
        record.files.raw_json_sha256, ...
        record.files.solution_csv_sha256, ...
        record.files.solution_model_sha256};
    if strcmp(record_status, 'complete')
        if ~strcmp(stage, 'simulation') || ~isempty(message)
            error('publish_comsol_batch_progress:CompleteCase', ...
                ['Complete records must be at simulation stage and have ' ...
                'an empty message.']);
        end
        require_hash(hashes{1}, false, 'raw_csv_sha256');
        require_hash(hashes{2}, false, 'raw_json_sha256');
        require_hash(hashes{3}, false, 'solution_csv_sha256');
        require_hash(hashes{4}, ~save_model, 'solution_model_sha256');
    elseif strcmp(record_status, 'failed')
        if ~strcmp(stage, 'simulation') || isempty(message)
            error('publish_comsol_batch_progress:FailedCase', ...
                ['Failed records must be at simulation stage and contain ' ...
                'a failure message.']);
        end
        require_empty_hashes(hashes);
    elseif strcmp(record_status, 'pending')
        if ~isempty(stage) || ~isempty(message)
            error('publish_comsol_batch_progress:PendingCase', ...
                'Pending records must have empty stage and message values.');
        end
        require_empty_hashes(hashes);
    else
        error('publish_comsol_batch_progress:CaseStatus', ...
            'Case status must be pending, complete, or failed.');
    end
end
end

function require_empty_hashes(hashes)
for hash_index = 1:numel(hashes)
    require_hash(hashes{hash_index}, true, 'non-complete case digest');
    if strlength(string(hashes{hash_index})) ~= 0
        error('publish_comsol_batch_progress:CaseDigest', ...
            'Pending and failed records cannot publish file digests.');
    end
end
end

function require_hash(value, allow_empty, label)
text = require_text(value, label, allow_empty);
if allow_empty && isempty(text)
    return;
end
if isempty(regexp(text, '^[0-9a-f]{64}$', 'once'))
    error('publish_comsol_batch_progress:CaseDigest', ...
        '%s must be a lowercase SHA-256 digest.', label);
end
end

function case_ids = validate_case_ids(value)
if isstring(value) && isvector(value)
    case_ids = value(:);
elseif iscell(value) && isvector(value)
    case_ids = strings(numel(value), 1);
    for case_index = 1:numel(value)
        case_ids(case_index) = string(require_text( ...
            value{case_index}, 'intended_case_ids', false));
    end
else
    error('publish_comsol_batch_progress:IntendedCases', ...
        'intended_case_ids must be a text vector.');
end
if isempty(case_ids) || numel(unique(case_ids)) ~= numel(case_ids)
    error('publish_comsol_batch_progress:IntendedCases', ...
        'intended_case_ids must be non-empty and unique.');
end
for case_index = 1:numel(case_ids)
    if isempty(regexp(char(case_ids(case_index)), ...
            '^case_[0-9]{4}$', 'once'))
        error('publish_comsol_batch_progress:IntendedCases', ...
            'intended_case_ids must use canonical case_0000 values.');
    end
end
end

function text = require_text(value, label, allow_empty)
if ischar(value) && (isrow(value) || isempty(value))
    text = value;
elseif isstring(value) && isscalar(value) && ~ismissing(value)
    text = char(value);
else
    error('publish_comsol_batch_progress:Text', ...
        '%s must be a text scalar.', label);
end
if ~allow_empty && isempty(text)
    error('publish_comsol_batch_progress:Text', ...
        '%s must be non-empty.', label);
end
end

function write_json_atomic(path, payload)
temp_path = [path '.tmp'];
if isfile(temp_path), delete(temp_path); end
fid = fopen(temp_path, 'w');
if fid < 0
    error('publish_comsol_batch_progress:Open', ...
        'Could not open temporary progress file: %s', temp_path);
end
try
    fprintf(fid, '%s', jsonencode(payload, 'PrettyPrint', true));
    fclose(fid);
catch write_error
    fclose(fid);
    if isfile(temp_path), delete(temp_path); end
    rethrow(write_error);
end
[move_ok, move_message] = movefile(temp_path, path, 'f');
if ~move_ok
    if isfile(temp_path), delete(temp_path); end
    error('publish_comsol_batch_progress:Publish', ...
        'Failed to publish batch progress: %s', move_message);
end
end
