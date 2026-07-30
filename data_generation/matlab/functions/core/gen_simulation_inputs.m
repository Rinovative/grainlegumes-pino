% ============================================================
% gen_simulation_inputs.m
%
% Wrapper generating structure, material and BC fields
% and delegating export to gen_export_fields.
%
% Author: Rino M. Albertin
% Date:   2026-01-03
% ============================================================

function [fields, info] = gen_simulation_inputs(Lx, Ly, res, seed, opts)
%% === Path setup =============================================
this_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(this_dir, 'gen'));

%% === Defaults ===============================================
if nargin < 4 || isempty(opts)
    opts = struct();
end

if ~isfield(opts,'save'), opts.save = true; end

if ~isfield(opts,'save_dir')
    generated_data_root = getenv('GENERATED_DATA_ROOT');
    if isempty(generated_data_root)
        generation_root = fullfile(this_dir, '..', '..', '..');
        generation_root = char(java.io.File(generation_root).getCanonicalPath());
        generated_data_root = fullfile(generation_root, 'data');
    end
    opts.save_dir = fullfile(generated_data_root, 'raw', 'test');
end

if ~isfield(opts,'file_tag'), opts.file_tag = ""; end

fields = struct();
info   = struct();

%% === 1) Structure field =====================================
[fields, info.structure] = gen_structure_field(Lx, Ly, res, seed, opts);
info.geometry = struct( ...
    'Lx', Lx, ...
    'Ly', Ly, ...
    'dx', res, ...
    'dy', res, ...
    'nx', size(fields.grid.X,2), ...
    'ny', size(fields.grid.X,1), ...
    'res', res ...
);

%% === 2) Permeability + tensor ===============================
[fields, info.permeability] = gen_permeability_field(fields, opts);

%% === 3) Porosity ============================================
[fields, info.porosity] = gen_porosity_field(fields, opts);

%% === 4) Pressure boundary condition =========================
[fields, info.bc] = gen_pressure_bc(fields, opts);

%% === 5) Export (CSV + JSON) =================================
if opts.save
    info.export = gen_export(fields, info, opts);
end

end
