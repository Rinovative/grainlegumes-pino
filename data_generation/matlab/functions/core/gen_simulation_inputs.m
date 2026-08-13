%% gen_simulation_inputs.m
% ============================================================
% Orchestrate reproducible field generation and optional case export.
%
% Author: Rino M. Albertin
% Date:   2026-08-13
%
% DESCRIPTION
%   Resolves the current version-1 generation contract and case-specific
%   packing seed, then generates the structure, permeability tensor, porosity,
%   and inlet-pressure boundary condition in their canonical order.
%
% USAGE
%   [fields, info] = gen_simulation_inputs(Lx, Ly, res, seed, opts)
%
% INPUTS
%   Lx, Ly
%       Positive physical domain dimensions [m].
%   res
%       Cartesian grid spacing [m].
%   seed
%       Spatial realization seed passed to gen_structure_field.
%   opts
%       Shared generator options. packing_batch_variation, packing_batch_seed,
%       and packing_case_id identify the packing realization. save controls
%       export; save_dir and file_tag select the output location and basename.
%
% OUTPUTS
%   fields
%       Grid, structure, material, and boundary-condition fields.
%   info
%       Geometry and per-stage metadata; includes export metadata when save=true.
%
% NOTES
%   Contract resolution and semantic packing-seed derivation occur before the
%   spatial RNG is initialized. The packing draw therefore remains independent
%   of random-number consumption in the spatial field pipeline.
% ============================================================

function [fields, info] = gen_simulation_inputs(Lx, Ly, res, seed, opts)

%% === Path setup =============================================
this_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(this_dir, 'gen'));

%% === Defaults ===============================================
if nargin < 5 || isempty(opts)
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
if ~isfield(opts,'packing_batch_variation')
    opts.packing_batch_variation = 0.8;
end
if ~isfield(opts,'packing_batch_seed')
    opts.packing_batch_seed = seed;
end
if ~isfield(opts,'packing_case_id')
    opts.packing_case_id = 1;
end

% Resolve the batch contract and semantic packing seed before any
% stochastic spatial field is generated.
[~, ~, opts.generation_contract] = ...
    sample_parameters('contract', opts.packing_batch_variation);
opts.packing_scatter_seed = derive_packing_scatter_seed( ...
    opts.generation_contract.generation_contract_version, ...
    opts.packing_batch_seed, opts.packing_case_id);

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

function seed = derive_packing_scatter_seed(version, batch_seed, case_id)
% Derive a stable local-stream seed for the fixed packing_scatter semantic.

if ~(isscalar(version) && version == 1)
    error('derive_packing_scatter_seed:InvalidVersion', ...
        'generation contract version must be exactly 1.');
end
if ~(isscalar(batch_seed) && isfinite(batch_seed) && batch_seed >= 0 && ...
        batch_seed <= 2^32 - 1 && floor(batch_seed) == batch_seed)
    error('derive_packing_scatter_seed:InvalidBatchSeed', ...
        'batch_seed must be an integer in [0, 2^32-1].');
end
if ~(isscalar(case_id) && isfinite(case_id) && case_id >= 1 && floor(case_id) == case_id)
    error('derive_packing_scatter_seed:InvalidCaseId', ...
        'case_id must be a positive integer.');
end
identity = sprintf('packing_scatter|v%.0f|batch%.0f|case%.0f', ...
    version, batch_seed, case_id);
modulus = 4294967291;
hash = 2166136261;
for index = 1:numel(identity)
    hash = mod(hash * 65599 + double(identity(index)), modulus);
end
seed = floor(hash);
if seed == 0
    seed = 1;
end
end
