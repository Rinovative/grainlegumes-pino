%% run_comsol_case.m
% ============================================================
% Run one COMSOL Darcy-Brinkman simulation from generated task fields.
%
% The seven input CSV columns have one explicit contract:
%   x [m], y [m], Kxx [m^2], Kxy [m^2], Kyy [m^2], eps [1], p_bc [Pa].
%
% The copied model contains one combined interpolation feature whose visible
% label is "int" and whose internal COMSOL feature tag is "int1". For each
% case, the generated CSV is imported into that feature, which exposes the
% functions int1 through int5.
%
% The MPH template owns the Brinkman permeability matrix, material porosity,
% and inlet-pressure bindings. This runner validates the required model nodes
% but does not rewrite those template settings.
%
% The final solution CSV is published only after a complete export. Working
% models and temporary exports are removed on both success and failure.
% ============================================================

function results = run_comsol_case(field_path, template_path, output_dir, save_model)

if nargin < 4
    save_model = false;
end

addpath('C:\Program Files\COMSOL64\mli');
import com.comsol.model.*
import com.comsol.model.util.*

t_start = tic;
field_path = char(field_path);
template_path = char(template_path);
output_dir = char(output_dir);

if ~isfile(field_path)
    error('run_comsol_case:MissingInputField', ...
        'Input field CSV does not exist: %s', field_path);
end
if ~isfile(template_path)
    error('run_comsol_case:MissingTemplate', ...
        'COMSOL template does not exist: %s', template_path);
end
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

[~, name, ~] = fileparts(field_path);
case_path = fullfile(output_dir, [name '.mph']);
export_csv = fullfile(output_dir, [name '_sol.csv']);
temp_export_csv = fullfile(output_dir, ['.' name '_sol.tmp.csv']);
solved_model_path = fullfile(output_dir, [name '_sol.mph']);
temp_solved_model_path = fullfile(output_dir, ['.' name '_sol.tmp.mph']);

if isfile(export_csv)
    error('run_comsol_case:ExistingExport', ...
        'Refusing to overwrite existing COMSOL export: %s', export_csv);
end
if save_model && isfile(solved_model_path)
    error('run_comsol_case:ExistingModel', ...
        'Refusing to overwrite existing solved model: %s', solved_model_path);
end

[x_values, y_values] = validate_input_field_file(field_path);

cleanup_file(temp_export_csv);
cleanup_file(temp_solved_model_path);
cleanup_file(case_path);

[copy_ok, copy_message] = copyfile(template_path, case_path);
if ~copy_ok
    error('run_comsol_case:CopyTemplate', ...
        'Failed to copy COMSOL template to working model: %s', copy_message);
end

try
    model = mphload(case_path);
catch load_error
    cleanup_file(case_path);
    rethrow(load_error);
end

model_tag = char(model.tag);
resource_cleanup = onCleanup(@() cleanup_case_resources( ...
    model_tag, case_path, temp_export_csv, temp_solved_model_path)); %#ok<NASGU>

configure_interpolation_functions(model, field_path);
validate_template_physics_contract(model);

solve_timer = tic;
mphrun(model);
comsol_solve_s = toc(solve_timer);

export_tags = java_string_array_to_cell(model.result.export.tags);
if any(strcmp(export_tags, 'data1'))
    exp_obj = model.result.export('data1');
else
    exp_obj = model.result.export.create('data1', 'Data');
end

exp_obj.set('data', 'dset1');
exp_obj.set('filename', temp_export_csv);
exp_obj.set('separator', ';');
exp_obj.set('expr', { ...
    'br.kappaxx', 'br.kappayx', ...
    'br.kappaxy', 'br.kappayy', ...
    'int4(x,y)', 'int5(x,y)', ...
    'p', 'u', 'v', 'br.U'});
exp_obj.set('unit', { ...
    'm^2', 'm^2', 'm^2', 'm^2', ...
    '1', 'Pa', 'Pa', 'm/s', 'm/s', 'm/s'});
exp_obj.set('location', 'grid');
exp_obj.set('gridstruct', 'spreadsheet');
exp_obj.set('gridx2', uniform_grid_expression(x_values, 'x'));
exp_obj.set('gridy2', uniform_grid_expression(y_values, 'y'));
exp_obj.set('header', 'on');
exp_obj.set('fullprec', 'on');
exp_obj.set('sort', 'on');
exp_obj.set('includecoords', true);
exp_obj.set('includenan', false);
exp_obj.run;

if ~isfile(temp_export_csv)
    error('run_comsol_case:MissingTemporaryExport', ...
        'COMSOL did not create the expected temporary export: %s', ...
        temp_export_csv);
end

if save_model
    mphsave(model, temp_solved_model_path);
    if ~isfile(temp_solved_model_path)
        error('run_comsol_case:MissingTemporaryModel', ...
            'COMSOL did not create the expected temporary model: %s', ...
            temp_solved_model_path);
    end
end

[move_ok, move_message] = movefile(temp_export_csv, export_csv);
if ~move_ok
    error('run_comsol_case:PublishExport', ...
        'Failed to publish COMSOL export: %s', move_message);
end

if save_model
    [move_ok, move_message] = movefile( ...
        temp_solved_model_path, solved_model_path);
    if ~move_ok
        cleanup_file(export_csv);
        error('run_comsol_case:PublishModel', ...
            ['Failed to publish solved model; rolled back the CSV export: ' ...
            '%s'], ...
            move_message);
    end
end

results = struct( ...
    'name', name, ...
    'field_path', field_path, ...
    'export_csv', export_csv, ...
    'save_model', save_model, ...
    'comsol_solve_s', comsol_solve_s, ...
    'time_s', toc(t_start));
end

function [x_values, y_values] = validate_input_field_file(field_path)
% Validate the canonical seven-column input CSV and return its grid axes.

field_data = readmatrix(field_path, 'Delimiter', ';');

if isempty(field_data) || size(field_data, 2) ~= 7
    error('run_comsol_case:InputColumnContract', ...
        ['Input CSV must contain exactly seven numeric columns in this order: ' ...
        'x [m], y [m], Kxx [m^2], Kxy [m^2], Kyy [m^2], eps [1], p_bc [Pa].']);
end
if any(~isfinite(field_data), 'all')
    error('run_comsol_case:NonfiniteInput', ...
        'Input CSV contains non-finite values: %s', field_path);
end
if any(field_data(:, 3) <= 0) || any(field_data(:, 5) <= 0)
    error('run_comsol_case:InvalidPermeability', ...
        'Kxx and Kyy must be strictly positive in %s.', field_path);
end
if any(field_data(:, 3) .* field_data(:, 5) - field_data(:, 4).^2 <= 0)
    error('run_comsol_case:InvalidPermeability', ...
        ['The symmetric 2-D permeability tensor must be positive definite ' ...
        'in %s.'], ...
        field_path);
end
if any(field_data(:, 6) <= 0 | field_data(:, 6) > 1)
    error('run_comsol_case:InvalidPorosity', ...
        'Porosity must satisfy 0 < eps <= 1 in %s.', field_path);
end

x_values = unique(field_data(:, 1), 'sorted');
y_values = unique(field_data(:, 2), 'sorted');
coordinate_pairs = unique(field_data(:, 1:2), 'rows');

if numel(x_values) * numel(y_values) ~= size(field_data, 1) || ...
        size(coordinate_pairs, 1) ~= size(field_data, 1)
    error('run_comsol_case:IncompleteGrid', ...
        ['Input coordinates must form one complete Cartesian grid without ' ...
        'duplicates.']);
end

validate_uniform_axis(x_values, 'x');
validate_uniform_axis(y_values, 'y');
end

function validate_uniform_axis(values, axis_name)
% Validate that one coordinate axis is strictly increasing and uniform.

if numel(values) < 2
    error('run_comsol_case:InvalidGrid', ...
        '%s-coordinate grid must contain at least two distinct points.', ...
        axis_name);
end

spacing = diff(values);
tolerance = 1e-9 * max(1, abs(mean(spacing)));

if any(spacing <= 0) || any(abs(spacing - mean(spacing)) > tolerance)
    error('run_comsol_case:NonuniformGrid', ...
        '%s-coordinate grid must be strictly increasing and uniform.', ...
        axis_name);
end
end

function expression = uniform_grid_expression(values, axis_name)
% Build the COMSOL range expression for one validated uniform grid axis.

validate_uniform_axis(values, axis_name);
expression = sprintf('range(%.17g[m],%.17g[m],%.17g[m])', ...
    values(1), mean(diff(values)), values(end));
end

function configure_interpolation_functions(model, field_path)
% Import the current case CSV into the combined template interpolation feature.

feature_tag = 'int1';
expected_functions = {'int1', 'int2', 'int3', 'int4', 'int5'};

available_tags = java_string_array_to_cell(model.func.tags);
if ~any(strcmp(available_tags, feature_tag))
    error('run_comsol_case:MissingInterpolationFeature', ...
        ['The COMSOL template is missing the required interpolation feature ' ...
        'with tag ''%s''. Available function-feature tags: %s'], ...
        feature_tag, strjoin(available_tags, ', '));
end

interpolation = model.func(feature_tag);
configured_functions = ...
    java_string_array_to_cell(interpolation.functionNames());

if ~isequal(configured_functions(:)', expected_functions)
    error('run_comsol_case:InterpolationFunctionContract', ...
        ['Interpolation feature ''%s'' must expose the functions %s. ' ...
        'Found: %s'], ...
        feature_tag, ...
        strjoin(expected_functions, ', '), ...
        strjoin(configured_functions, ', '));
end

% Preserve the template-defined argument columns, value-column mappings,
% units, interpolation method, and extrapolation behavior. Only replace the
% source CSV for the current case and reload its data.
interpolation.set('filename', field_path);
interpolation.importData();

imported_functions = ...
    java_string_array_to_cell(interpolation.functionNames());

if ~isequal(imported_functions(:)', expected_functions)
    error('run_comsol_case:InterpolationImportContract', ...
        ['After importing the current CSV, interpolation feature ''%s'' ' ...
        'did not expose the expected functions. Expected: %s. Found: %s'], ...
        feature_tag, ...
        strjoin(expected_functions, ', '), ...
        strjoin(imported_functions, ', '));
end
end

function validate_template_physics_contract(model)
% Validate the required template-owned physics and material nodes.
%
% The MPH template owns the detailed permeability matrix, material porosity,
% and inlet-pressure expressions. This check deliberately avoids rewriting or
% serializing those settings because their API representation can differ
% between COMSOL releases.

try
    component = model.component('comp1');
    brinkman = component.physics('br');

    brinkman.feature('porous1').feature('pm1');
    brinkman.feature('inl1');
    component.material('mat1');
catch binding_error
    error('run_comsol_case:TemplatePhysicsContract', ...
        ['Template must contain comp1/br/porous1/pm1, comp1/br/inl1, ' ...
        'and material mat1: %s'], ...
        binding_error.message);
end
end

function values = java_string_array_to_cell(raw_values)
% Convert COMSOL string arrays to MATLAB character-vector cells.

if iscell(raw_values)
    values = cellfun(@char, raw_values(:)', 'UniformOutput', false);
    return;
end
if isstring(raw_values)
    values = cellstr(raw_values(:)');
    return;
end
if ischar(raw_values)
    values = {raw_values};
    return;
end
if isa(raw_values, 'java.lang.String')
    values = {char(raw_values)};
    return;
end
if ~isjava(raw_values)
    error('run_comsol_case:StringArrayConversion', ...
        'Expected a MATLAB or Java string array, received %s.', ...
        class(raw_values));
end

n_values = java.lang.reflect.Array.getLength(raw_values);
values = cell(1, n_values);

for value_index = 1:n_values
    java_value = java.lang.reflect.Array.get(raw_values, value_index - 1);
    values{value_index} = char(java_value);
end
end

function cleanup_case_resources(model_tag, case_path, temp_export_path, ...
        temp_model_path)
% Remove the COMSOL model before deleting its temporary files.

cleanup_model(model_tag);
cleanup_file(temp_export_path);
cleanup_file(temp_model_path);
cleanup_file(case_path);
end

function cleanup_file(path)
% Delete a temporary file idempotently, retrying brief COMSOL file locks.

if ~isfile(path)
    return;
end

last_message = '';

for attempt = 1:3
    try
        javaMethod('deleteIfExists', 'java.nio.file.Files', ...
            java.io.File(path).toPath());

        if ~isfile(path)
            return;
        end
    catch cleanup_error
        last_message = cleanup_error.message;
    end

    if attempt < 3
        pause(0.1);
    end
end

if isempty(last_message)
    last_message = 'File still exists after all cleanup attempts.';
end

warning('run_comsol_case:CleanupFile', ...
    'Could not remove temporary file %s: %s', path, last_message);
end

function cleanup_model(model_tag)
% Remove the case-owned COMSOL model from the connected server.

import com.comsol.model.util.*

if isempty(model_tag)
    return;
end

try
    ModelUtil.remove(model_tag);
catch cleanup_error
    warning('run_comsol_case:CleanupModel', ...
        'Could not remove COMSOL model %s: %s', ...
        model_tag, cleanup_error.message);
end
end
