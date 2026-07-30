function results = repair_comsol_case_timing( ...
        field_path, template_path, processed_dir, save_model, runner)
%REPAIR_COMSOL_CASE_TIMING Remeasure one solve without touching canonical output.
% The existing raw input CSV is solved through run_comsol_case in a private
% sibling directory. The private result proves the full solve/export lifecycle
% succeeded, but it is discarded so manifest-authoritative scientific outputs
% remain byte-for-byte unchanged.

if nargin < 5
    runner = @run_comsol_case;
end
field_path = require_existing_file(field_path, 'field_path');
template_path = require_existing_file(template_path, 'template_path');
processed_dir = require_directory(processed_dir, 'processed_dir');
if ~islogical(save_model) || ~isscalar(save_model)
    error('repair_comsol_case_timing:SaveModel', ...
        'save_model must be a logical scalar.');
end
if ~isa(runner, 'function_handle') || ~isscalar(runner)
    error('repair_comsol_case_timing:Runner', ...
        'runner must be a scalar function handle.');
end

[~, case_id, extension] = fileparts(field_path);
if ~strcmp(extension, '.csv') || ...
        isempty(regexp(case_id, '^case_[0-9]{4}$', 'once'))
    error('repair_comsol_case_timing:CaseId', ...
        'field_path must name a canonical case_0000.csv input.');
end
canonical_csv = fullfile(processed_dir, [case_id '_sol.csv']);
canonical_model = fullfile(processed_dir, [case_id '_sol.mph']);
canonical_csv_digest = sha256_required(canonical_csv, 'solution CSV');
if save_model
    canonical_model_digest = sha256_required( ...
        canonical_model, 'solution model');
else
    canonical_model_digest = '';
end
raw_field_digest = sha256_required(field_path, 'raw input CSV');
raw_metadata_path = fullfile(fileparts(field_path), [case_id '.json']);
raw_metadata_digest = sha256_required(raw_metadata_path, 'raw metadata JSON');

private_output_dir = fullfile( ...
    processed_dir, ['.' case_id '_timing_repair']);
if isfile(private_output_dir)
    error('repair_comsol_case_timing:PrivateDirectory', ...
        'Private timing-repair path is unexpectedly a file: %s', ...
        private_output_dir);
end
cleanup_private_directory(private_output_dir);
[mkdir_ok, mkdir_message] = mkdir(private_output_dir);
if ~mkdir_ok
    error('repair_comsol_case_timing:PrivateDirectory', ...
        'Could not create private timing-repair directory: %s', ...
        mkdir_message);
end
private_cleanup = onCleanup( ...
    @() cleanup_private_directory_best_effort( ...
        private_output_dir)); %#ok<NASGU>

try
    results = runner( ...
        field_path, template_path, private_output_dir, save_model);
    if ~isstruct(results) || ~isscalar(results) || ...
            ~isfield(results, 'comsol_solve_s') || ...
            ~is_positive_duration(results.comsol_solve_s)
        error('repair_comsol_case_timing:TimingResult', ...
            ['The private COMSOL solve did not return a valid ' ...
            'solve duration.']);
    end
    private_csv = fullfile(private_output_dir, [case_id '_sol.csv']);
    if ~isfile(private_csv)
        error('repair_comsol_case_timing:PrivateOutput', ...
            ['The private timing-repair solve did not publish its ' ...
            'solution CSV.']);
    end
    if save_model && ...
            ~isfile(fullfile(private_output_dir, [case_id '_sol.mph']))
        error('repair_comsol_case_timing:PrivateOutput', ...
            ['The private timing-repair solve did not publish its ' ...
            'solved model.']);
    end
catch repair_error
    try
        verify_all_unchanged( ...
            field_path, raw_field_digest, raw_metadata_path, ...
            raw_metadata_digest, canonical_csv, canonical_csv_digest, ...
            canonical_model, canonical_model_digest, save_model);
    catch integrity_error
        rethrow(integrity_error);
    end
    rethrow(repair_error);
end
verify_all_unchanged( ...
    field_path, raw_field_digest, raw_metadata_path, ...
    raw_metadata_digest, canonical_csv, canonical_csv_digest, ...
    canonical_model, canonical_model_digest, save_model);

end

function verify_all_unchanged( ...
        field_path, raw_field_digest, raw_metadata_path, ...
        raw_metadata_digest, canonical_csv, canonical_csv_digest, ...
        canonical_model, canonical_model_digest, save_model)
verify_unchanged(field_path, raw_field_digest, 'raw input CSV');
verify_unchanged(raw_metadata_path, raw_metadata_digest, ...
    'raw metadata JSON');
verify_unchanged(canonical_csv, canonical_csv_digest, ...
    'canonical solution CSV');
if save_model
    verify_unchanged(canonical_model, canonical_model_digest, ...
        'canonical solution model');
end
end

function path = require_existing_file(value, label)
path = require_text(value, label);
if ~isfile(path)
    error('repair_comsol_case_timing:MissingFile', ...
        '%s does not exist: %s', label, path);
end
end

function path = require_directory(value, label)
path = require_text(value, label);
if ~isfolder(path)
    error('repair_comsol_case_timing:MissingDirectory', ...
        '%s does not exist: %s', label, path);
end
end

function text = require_text(value, label)
if ischar(value) && isrow(value) && ~isempty(value)
    text = value;
elseif isstring(value) && isscalar(value) && ...
        ~ismissing(value) && strlength(value) > 0
    text = char(value);
else
    error('repair_comsol_case_timing:Text', ...
        '%s must be a non-empty text scalar.', label);
end
end

function verify_unchanged(path, expected_digest, label)
if ~isfile(path) || ~strcmp(sha256_file(path), expected_digest)
    error('repair_comsol_case_timing:CanonicalMutation', ...
        '%s changed during private timing repair: %s', label, path);
end
end

function digest = sha256_required(path, label)
if ~isfile(path)
    error('repair_comsol_case_timing:MissingCanonicalOutput', ...
        'Timing repair requires the existing canonical %s: %s', ...
        label, path);
end
digest = sha256_file(path);
end

function digest = sha256_file(path)
message_digest = java.security.MessageDigest.getInstance('SHA-256');
file_bytes = java.nio.file.Files.readAllBytes(java.io.File(path).toPath());
digest_bytes = typecast(int8(message_digest.digest(file_bytes)), 'uint8');
digest = lower(reshape(dec2hex(digest_bytes, 2).', 1, []));
end

function tf = is_positive_duration(value)
tf = isnumeric(value) && ~islogical(value) && isreal(value) && ...
    isscalar(value) && isfinite(value) && value > 0;
end

function cleanup_private_directory_best_effort(path)
if isfolder(path)
    [~, ~] = rmdir(path, 's');
end
end

function cleanup_private_directory(path)
if isfolder(path)
    rmdir(path, 's');
end
end
