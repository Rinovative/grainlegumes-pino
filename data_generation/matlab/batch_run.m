%% batch_run.m
% Start COMSOL server manually via:
%   "C:\Program Files\COMSOL63\bin\win64\comsolmphserver.exe"
%
% ============================================================
% Full batch pipeline: parameter sampling → κ(x,y) generation → COMSOL simulation
% Author: Rino M. Albertin
% Date: 2025-10-16
%
% DESCRIPTION
%   Executes a complete batch of synthetic permeability-field simulations:
%     1. Generate parameter samples (sample_parameters.m)
%     2. Generate κ(x,y) fields (gen_simulation_inputs.m)
%     3. Run Darcy–Brinkman COMSOL simulations (run_comsol_case.m)
%
%   Output structure:
%       <GENERATED_DATA_ROOT>/raw/<batch_name>/       → generated κ(x,y) fields
%       <GENERATED_DATA_ROOT>/processed/<batch_name>/ → exported COMSOL results
%
% REQUIREMENTS
%   • COMSOL Multiphysics with LiveLink for MATLAB
%   • Functions:
%       - sample_parameters.m
%       - gen_simulation_inputs.m
%       - run_comsol_case.m
% ============================================================

clear; clc;

%% --- Settings ----------------------------------------------------------
debug = false;             % unattended automation is the default
if debug
    dbstop if error
end

%% --- Configuration -----------------------------------------------------
this_file  = mfilename('fullpath');
script_dir = fileparts(this_file);

project_root = fullfile(script_dir, '..');
project_root = char(java.io.File(project_root).getCanonicalPath());
addpath(genpath(fullfile(project_root, 'matlab', 'functions')));

generated_data_root = getenv('GENERATED_DATA_ROOT');
if isempty(generated_data_root)
    storage_root = getenv('STORAGE_ROOT');
    if isempty(storage_root)
        repository_root = fileparts(project_root);
        storage_root = fullfile(fileparts(repository_root), 'storage');
    end
    generated_data_root = fullfile(storage_root, 'data_generation');
end

% === SAVE COMSOL WITH SOLUTION ===
save_model = false;

% === SAMPLING PARAMETERS ===
method     = 'sobol';       % 'uniform', 'lhs', or 'sobol'
variation  = 0.8;         % relative parameter variation
N          = 1000;        % number of samples
seed       = 13001;        % reproducibility seed

batch_name = sprintf('%s_var%.0f_seed%.0f', ...
    method, variation*100, seed);

% === PATHS ===
meta_dir      = fullfile(generated_data_root, 'meta');
raw_dir       = fullfile(generated_data_root, 'raw', batch_name);
processed_dir = fullfile(generated_data_root, 'processed', batch_name);
template_path = fullfile(project_root, 'comsol', 'template_brinkman.mph');

if ~isfolder(meta_dir), mkdir(meta_dir); end
if ~isfolder(raw_dir), mkdir(raw_dir); end
if ~isfolder(processed_dir), mkdir(processed_dir); end

%% --- Generate or load parameter samples -------------------------------
sample_csv = fullfile(meta_dir, [batch_name '.csv']);
sample_json = fullfile(meta_dir, [batch_name '.json']);

if ~isfile(sample_csv) && ~isfile(sample_json)
    disp("🧩 No existing sample found — generating new parameter set...");
    sample_parameters(method, variation, N, seed, meta_dir);
elseif xor(isfile(sample_csv), isfile(sample_json))
    error('batch_run:IncompleteSampleIdentity', ...
        'Sample CSV and JSON identity files must either both exist or both be absent.');
else
    disp("📂 Validating existing parameter sample identity.");
end
validate_sample_identity(sample_csv, sample_json, method, variation, N, seed);

%% --- COMSOL Connection -------------------------------------------------
addpath('C:\Program Files\COMSOL63\mli');
try
    v = mphversion;
    disp("✅ Connected to COMSOL Server: " + v);
catch
    disp("🔄 Connecting to COMSOL Server on port 2036...");
    mphstart(2036);
    pause(2);
    v = mphversion;
    disp("✅ Connected to COMSOL Server: " + v);
end

%% --- Load sample parameters -------------------------------------------
T = readtable(sample_csv, 'Delimiter', ';');
n_cases = height(T);
if n_cases ~= N || ~isequal(T.case_id, (1:N)')
    error('batch_run:SampleMembership', ...
        'Sample table must contain exactly case_id 1 through N in order.');
end
if ismember('simulate', T.Properties.VariableNames)
    simulate_mask = logical(T.simulate);
else
    simulate_mask = true(n_cases, 1);
end
intended_rows = find(simulate_mask);
intended_case_ids = arrayfun(@(case_id) sprintf('case_%04d', case_id), ...
    T.case_id(intended_rows), 'UniformOutput', false)';

disp("------------------------------------------------------------");
disp("🚀 Starting batch with " + n_cases + " cases (" + batch_name + ")");
disp("------------------------------------------------------------");

%% --- Fixed geometry parameters ----------------------------------------
Lx = 1.2;
Ly = 0.75;
res = 0.003;

manifest_path = fullfile(raw_dir, 'batch_manifest.json');
manifest_configuration = struct( ...
    'method', method, ...
    'variation', variation, ...
    'N', N, ...
    'seed', seed, ...
    'Lx', Lx, ...
    'Ly', Ly, ...
    'res', res, ...
    'save_model', save_model, ...
    'sample_sha256', sha256_file(sample_csv), ...
    'template_name', string(get_file_name(template_path)), ...
    'template_sha256', sha256_file(template_path));
manifest_field_schema = batch_field_schema();
validate_manifest_configuration(manifest_configuration);
prior_manifest_complete = validate_existing_manifest_identity( ...
    manifest_path, batch_name, manifest_configuration, ...
    manifest_field_schema, intended_case_ids, raw_dir, processed_dir);
if isfile(manifest_path)
    delete(manifest_path);
end
empty_file_hashes = struct( ...
    'raw_csv_sha256', "", ...
    'raw_json_sha256', "", ...
    'solution_csv_sha256', "", ...
    'solution_model_sha256', "");
case_records = repmat(struct( ...
    'case_id', "", ...
    'status', "pending", ...
    'stage', "", ...
    'message', "", ...
    'files', empty_file_hashes), numel(intended_rows), 1);
for record_index = 1:numel(intended_rows)
    case_records(record_index).case_id = string(intended_case_ids{record_index});
end

%% --- Start total timer -------------------------------------------------
t_batch_start = tic;
failures = strings(0, 1);

%% --- Main batch loop ---------------------------------------------------
for i = 1:n_cases
    case_id  = T.case_id(i);
    case_tag = sprintf('case_%04d', case_id);

    if ~simulate_mask(i)
        fprintf('[%4d/%4d] ⏭️  Meta-only Sobol case (no COMSOL): %s\n', ...
            i, n_cases, case_tag);
        continue;
    end
    record_index = find(intended_rows == i, 1);

    % ============================================================
    % RESUME LOGIC
    % A case is complete only when every configured final output exists.
    % Partial outputs are removed and recomputed through the atomic runner.
    % ============================================================
    sol_file = fullfile(processed_dir, sprintf('%s_sol.csv', case_tag));
    solved_model_file = fullfile(processed_dir, sprintf('%s_sol.mph', case_tag));
    working_model_file = fullfile(processed_dir, sprintf('%s.mph', case_tag));
    raw_field_file = fullfile(raw_dir, sprintf('%s.csv', case_tag));
    raw_metadata_file = fullfile(raw_dir, sprintf('%s.json', case_tag));
    outputs_complete = prior_manifest_complete && ...
        isfile(raw_field_file) && isfile(raw_metadata_file) && ...
        isfile(sol_file) && (~save_model || isfile(solved_model_file));
    if outputs_complete
        fprintf('[%4d/%4d] ⏩ Skip: configured outputs already exist (%s)\n', ...
            i, n_cases, case_tag);
        case_records(record_index).status = "complete";
        case_records(record_index).stage = "simulation";
        case_records(record_index).files = case_file_hashes( ...
            case_tag, raw_dir, processed_dir, save_model);
        continue;
    end
    if isfile(sol_file), delete(sol_file); end
    if isfile(solved_model_file), delete(solved_model_file); end
    if isfile(working_model_file), delete(working_model_file); end

    % --- Build option struct -------------------------------------------
    opts = struct( ...
        'k_mean',            T.k_mean(i), ...
        'var_rel',           T.var_rel(i), ...
        'base_len_rel',      T.base_len_rel(i), ...
        'smooth_len_rel',    T.smooth_len_rel(i), ...
        'ms_weight',         [T.msW_c(i), T.msW_f(i)], ...
        'anisotropy',        [T.ani_x(i), T.ani_y(i)], ...
        'coupling',          T.coupling(i), ...
        'noise_level',       T.noise_level(i), ...
        'noise_granularity', T.noise_granularity(i), ...
        'noise_bias',        T.noise_bias(i), ...
        'a_max',             T.a_max(i), ...
        'a_gamma',           T.a_gamma(i), ...
        'tensor_strength',   T.tensor_strength(i), ...
        'theta_jitter',      T.theta_jitter(i), ...
        'theta_smooth_rel',  T.theta_smooth_rel(i), ...
        ...
        'A_rel',             T.A_rel(i), ...
        'eps_smooth_rel',    T.eps_smooth_rel(i), ...
        'texture_amp',       T.texture_amp(i), ...
        ...
        'p_inlet_mean',      T.p_inlet_mean(i), ...
        'a_sin',             T.a_sin(i), ...
        'f_sin',             T.f_sin(i), ...
        'phi_sin',           T.phi_sin(i), ...
        'k_gauss',           T.k_gauss(i), ...
        'a_gauss',           T.a_gauss(i), ...
        'sigma_gauss',       T.sigma_gauss(i), ...
        'gauss_jitter',      T.gauss_jitter(i), ...
        'a_lin',             T.a_lin(i), ...
        ...
        'save',              true, ...
        'save_dir',          raw_dir, ...
        'file_tag',          case_tag ...
    );

    seed_case    = seed + case_id;

    %% --- Debug info ----------------------------------------------------
    if debug
        fprintf('\n[DEBUG] Case %d/%d (%s)\n', i, n_cases, case_tag);

        % --- global -----------------------------------------------------
        fprintf('  global: k_mean=%.2e | var_rel=%.2f | seed=%d\n', ...
            opts.k_mean, opts.var_rel, seed_case);

        % --- background -------------------------------------------------
        fprintf('  background: base_len=%.3f | smooth_len=%.3f | ms_weight=[%.2f %.2f] | anisotropy=[%.2f %.2f] | coupling=%.2f\n', ...
            opts.base_len_rel, opts.smooth_len_rel, ...
            opts.ms_weight(1), opts.ms_weight(2), ...
            opts.anisotropy(1), opts.anisotropy(2), ...
            opts.coupling);

        % --- noise ------------------------------------------------------
        fprintf('  noise: noise_level=%.2f | noise_granularity=%.2f | noise_bias=%.2f\n', ...
            opts.noise_level, opts.noise_granularity, opts.noise_bias);

        % --- tensor -----------------------------------------------------
        fprintf('  tensor: a_max=%.2f | a_gamma=%.2f | tensor_strength=%.2f | theta_jitter=%.3f | theta_smooth=%.3f\n', ...
            opts.a_max, opts.a_gamma, opts.tensor_strength, ...
            opts.theta_jitter, opts.theta_smooth_rel);

        % --- porosity ---------------------------------------------------
        fprintf('  porosity: A_rel=%.3f | eps_smooth_rel=%.3f | texture_amp=%.4f\n', ...
            opts.A_rel, opts.eps_smooth_rel, opts.texture_amp);

        % --- pressure BC --------------------------------------------
        fprintf('  p_bc: mean=%.1f | a_sin=%.3f | f_sin=%.2f | phi_sin=%.2f\n', ...
            opts.p_inlet_mean, opts.a_sin, opts.f_sin, opts.phi_sin);

        fprintf('  k_gauss=%d | a_gauss=%.3f | sigma_gauss=%.3f | jitter=%.2f | a_lin=%.3f\n', ...
            opts.k_gauss, opts.a_gauss, opts.sigma_gauss, opts.gauss_jitter, opts.a_lin);

        % --- io ---------------------------------------------------------
        fprintf('\n  io: save_dir=%s | file_tag=%s\n', ...
            opts.save_dir, opts.file_tag);
    end

    %% --- Step 1: Generate permeability field ---------------------------
    try
        [fields, info] = gen_simulation_inputs(Lx, Ly, res, seed_case, opts);
        if debug
            fprintf('  → Fields exported: %s\n', info.export.paths.csv);
        end
    catch ME
        fprintf('[%4d/%4d] ❌ Error in gen_simulation_inputs: %s\n', ...
            i, n_cases, ME.message);
        failures(end + 1, 1) = sprintf('%s generation: %s', ...
            case_tag, ME.message);
        case_records(record_index).status = "failed";
        case_records(record_index).stage = "generation";
        case_records(record_index).message = string(ME.message);
        continue;
    end

    %% --- Step 2: Run COMSOL simulation --------------------------------
    field_path = info.export.paths.csv;

    try
        results = run_comsol_case(field_path, template_path, processed_dir, save_model);
        if ~isfile(sol_file) || (save_model && ~isfile(solved_model_file))
            error('batch_run:MissingConfiguredOutput', ...
                'Runner returned without publishing every configured output.');
        end
        case_records(record_index).status = "complete";
        case_records(record_index).stage = "simulation";
        case_records(record_index).files = case_file_hashes( ...
            case_tag, raw_dir, processed_dir, save_model);
        fprintf('[%4d/%4d] ✅ COMSOL completed: %s (%.1f s)\n', ...
            i, n_cases, opts.file_tag, results.time_s);
    catch ME
        fprintf('[%4d/%4d] ❌ Error in COMSOL: %s\n', ...
            i, n_cases, ME.message);
        failures(end + 1, 1) = sprintf('%s COMSOL: %s', ...
            case_tag, ME.message);
        case_records(record_index).status = "failed";
        case_records(record_index).stage = "simulation";
        case_records(record_index).message = string(ME.message);
        continue;
    end
end

%% --- End total timer ---------------------------------------------------
t_batch_end = toc(t_batch_start);
t_min = t_batch_end / 60;
t_hr  = t_batch_end / 3600;

disp("------------------------------------------------------------");
if isempty(failures)
    manifest_status = "complete";
else
    manifest_status = "failed";
end
manifest = struct( ...
    'schema_kind', "comsol_batch_manifest", ...
    'schema_version', 1, ...
    'batch_name', string(batch_name), ...
    'status', manifest_status, ...
    'configuration', manifest_configuration, ...
    'field_schema', manifest_field_schema, ...
    'intended_case_ids', {intended_case_ids}, ...
    'cases', case_records);
write_json_atomic(manifest_path, manifest);

if isempty(failures)
    fprintf("🏁 Batch completed successfully.\n");
else
    fprintf("❌ Batch finished with %d failed case(s):\n", numel(failures));
    for failure_index = 1:numel(failures)
        fprintf("  - %s\n", failures(failure_index));
    end
end
fprintf("⏱️ Total time: %.1f s (%.2f min | %.2f h)\n", ...
    t_batch_end, t_min, t_hr);
disp("------------------------------------------------------------");

if debug
    dbclear if error
end
if ~isempty(failures)
    error('batch_run:CaseFailures', ...
        '%d case(s) failed; the generated batch is incomplete.', ...
        numel(failures));
end

function validate_sample_identity(sample_csv, sample_json, method, variation, N, seed)
if ~isfile(sample_csv) || ~isfile(sample_json)
    error('batch_run:MissingSampleIdentity', ...
        'Parameter sampling did not publish both CSV and JSON identity files.');
end
payload = jsondecode(fileread(sample_json));
if ~isstruct(payload) || ~isfield(payload, 'meta') || ...
        ~isstruct(payload.meta) || ~isfield(payload, 'n_cases')
    error('batch_run:SampleIdentity', ...
        'Sample JSON does not contain the required meta and n_cases contract.');
end
meta = payload.meta;
required = {'method', 'variation', 'N', 'seed'};
if ~all(isfield(meta, required)) || ...
        ~strcmpi(char(meta.method), method) || ...
        meta.variation ~= variation || meta.N ~= N || ...
        meta.seed ~= seed || payload.n_cases ~= N
    error('batch_run:SampleIdentity', ...
        ['Existing sample identity does not match method/variation/N/seed. ' ...
        'Use a distinct batch identity or remove the mismatched sample pair.']);
end
end

function complete = validate_existing_manifest_identity( ...
        path, batch_name, configuration, field_schema, intended_case_ids, ...
        raw_dir, processed_dir)
complete = false;
if ~isfile(path)
    return;
end
payload = jsondecode(fileread(path));
require_exact_struct_fields(payload, { ...
    'schema_kind', 'schema_version', 'batch_name', 'status', ...
    'configuration', 'field_schema', 'intended_case_ids', 'cases'}, ...
    'batch_run:ManifestIdentity', 'batch manifest');
if ~strcmp(require_text_scalar(payload.schema_kind, ...
        'batch_run:ManifestIdentity', 'schema_kind'), ...
        'comsol_batch_manifest') || ...
        ~is_real_numeric_scalar(payload.schema_version) || ...
        payload.schema_version ~= 1 || ...
        ~strcmp(require_text_scalar(payload.batch_name, ...
        'batch_run:ManifestIdentity', 'batch_name'), batch_name)
    error('batch_run:ManifestIdentity', ...
        'Existing batch manifest has an invalid schema or batch identity.');
end
validate_manifest_configuration(payload.configuration);
if ~manifest_configurations_equal(payload.configuration, configuration)
    error('batch_run:ManifestIdentity', ...
        ['Existing batch manifest belongs to a different configuration. ' ...
        'Refusing to reuse the same batch directory.']);
end
validate_manifest_field_schema(payload.field_schema, field_schema);
manifest_case_ids = require_string_vector(payload.intended_case_ids, ...
    'batch_run:ManifestIdentity', 'intended_case_ids');
if ~isequal(manifest_case_ids, string(intended_case_ids(:)))
    error('batch_run:ManifestIdentity', ...
        'Existing batch manifest has different intended case membership.');
end
status = require_text_scalar(payload.status, ...
    'batch_run:ManifestIdentity', 'status');
if ~ismember(status, {'complete', 'failed'})
    error('batch_run:ManifestIdentity', ...
        'Batch manifest status must be either complete or failed.');
end
if strcmp(status, 'complete')
    validate_complete_case_records(payload.cases, intended_case_ids, ...
        raw_dir, processed_dir, logical(configuration.save_model));
    complete = true;
end
end

function schema = batch_field_schema()
schema = struct( ...
    'input_columns', {{ ...
        'x', 'y', 'Kxx', 'Kxy', 'Kyy', 'eps', 'p_bc'}}, ...
    'solution_columns', {{ ...
        'x', 'y', 'kappaxx', 'kappayx', 'kappaxy', 'kappayy', ...
        'eps', 'p_bc', 'p', 'u', 'v', 'U'}});
end

function validate_manifest_configuration(configuration)
require_exact_struct_fields(configuration, { ...
    'method', 'variation', 'N', 'seed', 'Lx', 'Ly', 'res', ...
    'save_model', 'sample_sha256', 'template_name', ...
    'template_sha256'}, ...
    'batch_run:ManifestConfiguration', 'manifest configuration');
method = require_text_scalar(configuration.method, ...
    'batch_run:ManifestConfiguration', 'method');
if ~ismember(method, {'uniform', 'lhs', 'sobol'})
    error('batch_run:ManifestConfiguration', ...
        'Manifest method must be uniform, lhs, or sobol.');
end
require_finite_range(configuration.variation, 0, inf, ...
    'variation', 'batch_run:ManifestConfiguration');
require_integer_range(configuration.N, 1, flintmax, ...
    'N', 'batch_run:ManifestConfiguration');
require_integer_range(configuration.seed, 0, 2^32 - 1, ...
    'seed', 'batch_run:ManifestConfiguration');
require_finite_range(configuration.Lx, 0, inf, ...
    'Lx', 'batch_run:ManifestConfiguration', false);
require_finite_range(configuration.Ly, 0, inf, ...
    'Ly', 'batch_run:ManifestConfiguration', false);
require_finite_range(configuration.res, 0, inf, ...
    'res', 'batch_run:ManifestConfiguration', false);
if configuration.res > min(configuration.Lx, configuration.Ly)
    error('batch_run:ManifestConfiguration', ...
        'Manifest res cannot exceed the shorter domain dimension.');
end
if ~islogical(configuration.save_model) || ...
        ~isscalar(configuration.save_model)
    error('batch_run:ManifestConfiguration', ...
        'Manifest save_model must be a logical scalar.');
end
validate_sha256_value(configuration.sample_sha256, false, ...
    'sample_sha256', 'batch_run:ManifestConfiguration');
validate_sha256_value(configuration.template_sha256, false, ...
    'template_sha256', 'batch_run:ManifestConfiguration');
template_name = require_text_scalar(configuration.template_name, ...
    'batch_run:ManifestConfiguration', 'template_name');
[template_folder, template_base, template_extension] = fileparts(template_name);
if ~isempty(template_folder) || isempty(template_base) || ...
        ~strcmp(template_extension, '.mph') || ...
        ~strcmp([template_base template_extension], template_name)
    error('batch_run:ManifestConfiguration', ...
        'Manifest template_name must be a basename ending in .mph.');
end
end

function validate_manifest_field_schema(actual, expected)
require_exact_struct_fields(actual, {'input_columns', 'solution_columns'}, ...
    'batch_run:ManifestFieldSchema', 'field_schema');
actual_input = require_string_vector(actual.input_columns, ...
    'batch_run:ManifestFieldSchema', 'field_schema.input_columns');
actual_solution = require_string_vector(actual.solution_columns, ...
    'batch_run:ManifestFieldSchema', 'field_schema.solution_columns');
if ~isequal(actual_input, string(expected.input_columns(:))) || ...
        ~isequal(actual_solution, string(expected.solution_columns(:)))
    error('batch_run:ManifestFieldSchema', ...
        'Existing batch manifest field_schema is not the production schema.');
end
end

function validate_complete_case_records(records, intended_case_ids, ...
        raw_dir, processed_dir, save_model)
expected_case_ids = string(intended_case_ids(:));
if isempty(expected_case_ids)
    if ~isempty(records)
        error('batch_run:ManifestRecordContract', ...
            'Complete manifest contains records for an empty intended batch.');
    end
    return;
end
if ~isstruct(records) || numel(records) ~= numel(expected_case_ids)
    error('batch_run:ManifestRecordContract', ...
        'Complete manifest must contain exactly one record per intended case.');
end
for record_index = 1:numel(expected_case_ids)
    record = records(record_index);
    require_exact_struct_fields(record, ...
        {'case_id', 'status', 'stage', 'message', 'files'}, ...
        'batch_run:ManifestRecordContract', 'case record');
    case_id = require_text_scalar(record.case_id, ...
        'batch_run:ManifestRecordContract', 'case record case_id');
    status = require_text_scalar(record.status, ...
        'batch_run:ManifestRecordContract', 'case record status');
    stage = require_text_scalar(record.stage, ...
        'batch_run:ManifestRecordContract', 'case record stage');
    message = require_text_scalar(record.message, ...
        'batch_run:ManifestRecordContract', 'case record message');
    if ~strcmp(case_id, char(expected_case_ids(record_index))) || ...
            ~strcmp(status, 'complete') || ...
            ~strcmp(stage, 'simulation') || ~isempty(message)
        error('batch_run:ManifestRecordContract', ...
            ['Complete manifest case records must be ordered, complete, ' ...
            'at simulation stage, and have an empty message.']);
    end
    require_exact_struct_fields(record.files, { ...
        'raw_csv_sha256', 'raw_json_sha256', ...
        'solution_csv_sha256', 'solution_model_sha256'}, ...
        'batch_run:ManifestRecordContract', 'case record files');
    raw_csv_sha256 = validate_sha256_value( ...
        record.files.raw_csv_sha256, false, 'raw_csv_sha256', ...
        'batch_run:ManifestRecordContract');
    raw_json_sha256 = validate_sha256_value( ...
        record.files.raw_json_sha256, false, 'raw_json_sha256', ...
        'batch_run:ManifestRecordContract');
    solution_csv_sha256 = validate_sha256_value( ...
        record.files.solution_csv_sha256, false, ...
        'solution_csv_sha256', 'batch_run:ManifestRecordContract');
    solution_model_sha256 = validate_sha256_value( ...
        record.files.solution_model_sha256, ~save_model, ...
        'solution_model_sha256', 'batch_run:ManifestRecordContract');
    case_tag = char(expected_case_ids(record_index));
    verify_manifest_file(fullfile(raw_dir, [case_tag '.csv']), ...
        raw_csv_sha256, 'raw CSV');
    verify_manifest_file(fullfile(raw_dir, [case_tag '.json']), ...
        raw_json_sha256, 'raw JSON');
    verify_manifest_file(fullfile(processed_dir, [case_tag '_sol.csv']), ...
        solution_csv_sha256, 'solution CSV');
    model_path = fullfile(processed_dir, [case_tag '_sol.mph']);
    if save_model
        verify_manifest_file(model_path, solution_model_sha256, ...
            'solution model');
    elseif ~isempty(solution_model_sha256) || isfile(model_path)
        error('batch_run:ManifestFileIntegrity', ...
            ['Manifest configured save_model=false but a model digest or ' ...
            'authoritative model file exists for %s.'], case_tag);
    end
end
end

function files = case_file_hashes(case_tag, raw_dir, processed_dir, save_model)
raw_csv_path = fullfile(raw_dir, [case_tag '.csv']);
raw_json_path = fullfile(raw_dir, [case_tag '.json']);
solution_csv_path = fullfile(processed_dir, [case_tag '_sol.csv']);
solution_model_path = fullfile(processed_dir, [case_tag '_sol.mph']);
files = struct( ...
    'raw_csv_sha256', hash_required_file(raw_csv_path, 'raw CSV'), ...
    'raw_json_sha256', hash_required_file(raw_json_path, 'raw JSON'), ...
    'solution_csv_sha256', ...
        hash_required_file(solution_csv_path, 'solution CSV'), ...
    'solution_model_sha256', "");
if save_model
    files.solution_model_sha256 = ...
        hash_required_file(solution_model_path, 'solution model');
elseif isfile(solution_model_path)
    error('batch_run:ManifestFileIntegrity', ...
        ['Manifest configured save_model=false but a solution model exists: ' ...
        '%s'], solution_model_path);
end
end

function digest = hash_required_file(path, label)
if ~isfile(path)
    error('batch_run:ManifestFileIntegrity', ...
        'Cannot publish manifest: required %s is missing: %s', label, path);
end
digest = string(sha256_file(path));
end

function verify_manifest_file(path, expected_sha256, label)
if ~isfile(path)
    error('batch_run:ManifestFileIntegrity', ...
        'Manifest-authoritative %s is missing: %s', label, path);
end
actual_sha256 = sha256_file(path);
if ~strcmp(actual_sha256, expected_sha256)
    error('batch_run:ManifestFileIntegrity', ...
        'Manifest-authoritative %s SHA-256 mismatch: %s', label, path);
end
end

function digest = validate_sha256_value(value, allow_empty, label, error_id)
digest = require_text_scalar(value, error_id, label);
if allow_empty && isempty(digest)
    return;
end
if isempty(regexp(digest, '^[0-9a-f]{64}$', 'once'))
    error(error_id, ...
        '%s must be a lowercase 64-character SHA-256 digest.', label);
end
end

function tf = manifest_configurations_equal(left, right)
text_fields = {'method', 'sample_sha256', 'template_name', 'template_sha256'};
numeric_fields = {'variation', 'N', 'seed', 'Lx', 'Ly', 'res'};
tf = left.save_model == right.save_model;
for field_index = 1:numel(text_fields)
    field = text_fields{field_index};
    tf = tf && strcmp(char(string(left.(field))), char(string(right.(field))));
end
for field_index = 1:numel(numeric_fields)
    field = numeric_fields{field_index};
    tf = tf && left.(field) == right.(field);
end
end

function require_exact_struct_fields(value, expected_fields, error_id, label)
if ~isstruct(value) || ~isscalar(value) || ...
        ~isequal(sort(fieldnames(value)), sort(expected_fields(:)))
    error(error_id, '%s does not have the exact required fields.', label);
end
end

function text = require_text_scalar(value, error_id, label)
if ischar(value) && (isrow(value) || isempty(value))
    text = value;
elseif isstring(value) && isscalar(value) && ~ismissing(value)
    text = char(value);
else
    error(error_id, '%s must be a text scalar.', label);
end
end

function values = require_string_vector(value, error_id, label)
if isempty(value) && ~ischar(value) && ~isstring(value) && ~iscell(value)
    values = strings(0, 1);
elseif ischar(value) && (isrow(value) || isempty(value))
    values = string(value);
elseif isstring(value) && isvector(value)
    values = value(:);
elseif iscell(value) && isvector(value)
    values = strings(numel(value), 1);
    for value_index = 1:numel(value)
        values(value_index) = string(require_text_scalar( ...
            value{value_index}, error_id, label));
    end
else
    error(error_id, '%s must be a vector of text values.', label);
end
values = values(:);
if any(ismissing(values))
    error(error_id, '%s cannot contain missing values.', label);
end
end

function tf = is_real_numeric_scalar(value)
tf = isnumeric(value) && ~islogical(value) && isreal(value) && ...
    isscalar(value) && isfinite(value);
end

function require_integer_range(value, minimum, maximum, label, error_id)
if ~is_real_numeric_scalar(value) || value ~= fix(value) || ...
        value < minimum || value > maximum
    error(error_id, ...
        '%s must be an integer in the range [%.0f, %.0f].', ...
        label, minimum, maximum);
end
end

function require_finite_range(value, minimum, maximum, label, error_id, ...
        minimum_inclusive)
if nargin < 6
    minimum_inclusive = true;
end
if ~is_real_numeric_scalar(value)
    error(error_id, '%s must be a finite real numeric scalar.', label);
end
valid_minimum = value >= minimum;
if ~minimum_inclusive
    valid_minimum = value > minimum;
end
if ~valid_minimum || value > maximum
    if minimum_inclusive
        relation = '>=';
    else
        relation = '>';
    end
    error(error_id, '%s must be finite and %s %.17g.', ...
        label, relation, minimum);
end
end

function name = get_file_name(path)
[~, base, extension] = fileparts(path);
name = [base extension];
end

function digest = sha256_file(path)
message_digest = java.security.MessageDigest.getInstance('SHA-256');
file_bytes = java.nio.file.Files.readAllBytes(java.io.File(path).toPath());
digest_bytes = typecast(int8(message_digest.digest(file_bytes)), 'uint8');
digest = lower(reshape(dec2hex(digest_bytes, 2).', 1, []));
end

function write_json_atomic(path, payload)
temp_path = [path '.tmp'];
if isfile(temp_path), delete(temp_path); end
fid = fopen(temp_path, 'w');
if fid < 0
    error('batch_run:ManifestWrite', ...
        'Could not open temporary manifest for writing: %s', temp_path);
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
    error('batch_run:ManifestPublish', ...
        'Failed to publish batch manifest: %s', move_message);
end
end
