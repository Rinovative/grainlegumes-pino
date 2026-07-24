%% run_comsol_case.m
% ============================================================
% Run one COMSOL Darcy-Brinkman simulation from generated task fields.
%
% The seven input CSV columns have one explicit contract:
%   x [m], y [m], Kxx [m^2], Kxy [m^2], Kyy [m^2], eps [1], p_bc [Pa].
% The copied model is configured programmatically with interpolation functions
% int1 through int5 and matching Brinkman permeability/porosity/inlet bindings.
%
% The final solution CSV is published only after a complete export. Working
% models and temporary exports are removed on both success and failure.
% ============================================================

function results = run_comsol_case(field_path, template_path, output_dir, save_model)

if nargin < 4
    save_model = false;
end

addpath('C:\Program Files\COMSOL63\mli');
import com.comsol.model.*
import com.comsol.model.util.*

t_start = tic;
field_path = char(field_path);
template_path = char(template_path);
output_dir = char(output_dir);

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

cleanup_file(temp_export_csv);
cleanup_file(temp_solved_model_path);
cleanup_file(case_path);
copyfile(template_path, case_path);
case_cleanup = onCleanup(@() cleanup_file(case_path)); %#ok<NASGU>
temp_export_cleanup = onCleanup(@() cleanup_file(temp_export_csv)); %#ok<NASGU>
temp_model_cleanup = onCleanup(@() cleanup_file(temp_solved_model_path)); %#ok<NASGU>

model = mphload(case_path);
model_tag = char(model.tag);
model_cleanup = onCleanup(@() cleanup_model(model_tag)); %#ok<NASGU>

[x_values, y_values] = validate_input_field_file(field_path);
configure_interpolation_functions(model, field_path);
configure_and_validate_physics_bindings(model);

mphrun(model);

if any(strcmp(model.result.export.tags, 'data1'))
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
        'COMSOL did not create the expected temporary export: %s', temp_export_csv);
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
    [move_ok, move_message] = movefile(temp_solved_model_path, solved_model_path);
    if ~move_ok
        cleanup_file(export_csv);
        error('run_comsol_case:PublishModel', ...
            'Failed to publish solved model; rolled back the CSV export: %s', ...
            move_message);
    end
end

results = struct( ...
    'name', name, ...
    'field_path', field_path, ...
    'export_csv', export_csv, ...
    'save_model', save_model, ...
    'time_s', toc(t_start));
end

function [x_values, y_values] = validate_input_field_file(field_path)
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
        'The symmetric 2-D permeability tensor must be positive definite in %s.', ...
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
        'Input coordinates must form one complete Cartesian grid without duplicates.');
end
validate_uniform_axis(x_values, 'x');
validate_uniform_axis(y_values, 'y');
end

function validate_uniform_axis(values, axis_name)
if numel(values) < 2
    error('run_comsol_case:InvalidGrid', ...
        '%s-coordinate grid must contain at least two distinct points.', axis_name);
end
spacing = diff(values);
tolerance = 1e-9 * max(1, abs(mean(spacing)));
if any(spacing <= 0) || any(abs(spacing - mean(spacing)) > tolerance)
    error('run_comsol_case:NonuniformGrid', ...
        '%s-coordinate grid must be strictly increasing and uniform.', axis_name);
end
end

function expression = uniform_grid_expression(values, axis_name)
validate_uniform_axis(values, axis_name);
expression = sprintf('range(%.17g[m],%.17g[m],%.17g[m])', ...
    values(1), mean(diff(values)), values(end));
end

function configure_interpolation_functions(model, field_path)
int_tags = {'int1', 'int2', 'int3', 'int4', 'int5'};
value_columns = [3, 4, 5, 6, 7];
function_units = {'m^2', 'm^2', 'm^2', '1', 'Pa'};
column_keys = arrayfun(@(index) sprintf('col%d', index), 1:7, ...
    'UniformOutput', false);
available_tags = model.func.tags;

for k = 1:numel(int_tags)
    ftag = int_tags{k};
    if ~any(strcmp(available_tags, ftag))
        model.func.create(ftag, 'Interpolation');
    end
    column_types = cell(1, 14);
    function_names = cell(1, 14);
    for column_index = 1:7
        offset = 2 * column_index - 1;
        column_types(offset:offset + 1) = {column_keys{column_index}, 'none'};
        function_names(offset:offset + 1) = {column_keys{column_index}, ...
            sprintf('unused%d', column_index)};
    end
    column_types(2) = {'arg'};
    column_types(4) = {'arg'};
    column_types(2 * value_columns(k)) = {'value'};
    function_names(2) = {[ftag 'a']};
    function_names(4) = {ftag};
    function_names(2 * value_columns(k)) = {ftag};

    interpolation = model.func(ftag);
    interpolation.set('source', 'file');
    interpolation.set('filename', field_path);
    interpolation.set('struct', 'spreadsheet');
    interpolation.set('filecolumns', 7);
    interpolation.set('columnKeys', column_keys);
    interpolation.set('columnType', column_types);
    interpolation.set('funcnames', function_names);
    interpolation.set('nargs', 2);
    interpolation.set('fununit', {function_units{k}});
    interpolation.set('argunit', {'m', 'm'});
    interpolation.set('interp', 'linear');
    interpolation.set('extrap', 'const');
    interpolation.importData;

    imported_names = cellstr(interpolation.functionNames());
    configured_names = cellstr(interpolation.getStringArray('funcnames'));
    if numel(imported_names) ~= 1 || ~strcmp(imported_names{1}, ftag) || ...
            ~isequal(configured_names(:)', function_names)
        error('run_comsol_case:InterpolationColumnMapping', ...
            ['Interpolation %s did not expose exactly %s(x,y). Required value ' ...
            'column is CSV column %d with unit %s.'], ...
            ftag, ftag, value_columns(k), function_units{k});
    end
end
end

function configure_and_validate_physics_bindings(model)
try
    brinkman = model.component('comp1').physics('br');
    porous_medium = brinkman.feature('porous1').feature('pm1');
    inlet = brinkman.feature('inl1');
    porous_medium.set('kappa_mat', 'userdef');
    porous_medium.set('kappa', { ...
        'int1(x,y)', 'int2(x,y)', '0'; ...
        'int2(x,y)', 'int3(x,y)', '0'; ...
        '0', '0', 'int1(x,y)'});
    porous_medium.set('epsilon_p_mat', 'userdef');
    porous_medium.set('epsilon_p', 'int4(x,y)');
    inlet.set('BoundaryCondition', 'Pressure');
    inlet.set('p0', 'int5(x,y)');

    kappa = cellstr(porous_medium.getStringArray('kappa'));
    porosity = char(porous_medium.getString('epsilon_p'));
    inlet_pressure = char(inlet.getString('p0'));
catch binding_error
    error('run_comsol_case:TemplatePhysicsContract', ...
        ['Template must contain comp1/br/porous1/pm1 and comp1/br/inl1 with ' ...
        'user-defined permeability, porosity, and pressure settings: %s'], ...
        binding_error.message);
end

kappa = regexprep(kappa, '\s+', '');
if numel(kappa) < 5 || ...
        ~strcmp(kappa{1}, 'int1(x,y)') || ...
        ~strcmp(kappa{2}, 'int2(x,y)') || ...
        ~strcmp(kappa{4}, 'int2(x,y)') || ...
        ~strcmp(kappa{5}, 'int3(x,y)') || ...
        ~strcmp(regexprep(porosity, '\s+', ''), 'int4(x,y)') || ...
        ~strcmp(regexprep(inlet_pressure, '\s+', ''), 'int5(x,y)')
    error('run_comsol_case:TemplatePhysicsContract', ...
        ['COMSOL physics binding read-back did not match int1=Kxx, int2=Kxy, ' ...
        'int3=Kyy, int4=eps, and int5=p_bc.']);
end
end

function cleanup_file(path)
if isfile(path)
    try
        delete(path);
    catch cleanup_error
        warning('run_comsol_case:CleanupFile', ...
            'Could not remove temporary file %s: %s', path, cleanup_error.message);
    end
end
end

function cleanup_model(model_tag)
import com.comsol.model.util.*
if isempty(model_tag)
    return;
end
try
    ModelUtil.remove(model_tag);
catch cleanup_error
    warning('run_comsol_case:CleanupModel', ...
        'Could not remove COMSOL model %s: %s', model_tag, cleanup_error.message);
end
end
