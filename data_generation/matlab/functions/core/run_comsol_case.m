%% run_comsol_case.m
% ============================================================
% Run a single COMSOL Darcy–Brinkman simulation using a permeability field
% Author: Rino M. Albertin
% Date: 2025-10-16
%
% DESCRIPTION
%   Executes one COMSOL Multiphysics simulation case based on a provided
%   2D permeability field κ(x,y) [m²]. The function:
%       1. Loads a template model (.mph)
%       2. Replaces the interpolation function with the specified κ-field
%       3. Runs the COMSOL study (Darcy–Brinkman)
%       4. Exports all solution fields (velocity, pressure, κ-tensor)
%       5. Optionally saves the solved model (.mph)
%
%   Designed for automated batch execution of permeability-field studies
%   within a data generation or DoE pipeline.
%
% ------------------------------------------------------------------------
% USAGE
%   [model, results] = run_comsol_case(field_path, template_path, ...
%                                      output_dir, save_model)
%
% INPUTS
%   field_path    – Path to κ(x,y) .csv file used for interpolation
%   template_path – Path to COMSOL template .mph model
%   output_dir    – Directory for output (.mph, .csv)
%   save_model    – (optional) Boolean flag to save solved .mph model
%                   Default = false
%
% OUTPUTS
%   model   – COMSOL model object (only active during runtime)
%   results – Struct containing:
%                 .name        Model name / base filename
%                 .field_path  Path to input permeability field
%                 .export_csv  Path to exported solution .csv
%                 .save_model  Boolean flag for model saving
%                 .time_s      Total runtime [s]
%                 .status      Execution status ('ok' or 'error')
%
% DEPENDENCIES
%   • COMSOL Multiphysics with MATLAB LiveLink
%   • Template model must contain an interpolation function with tag 'int1'
%   • 'data1' export definition is reused or created automatically
%
% EXAMPLE
%   [model, res] = run_comsol_case('kappa_field.csv', ...
%                                  'template_brinkman.mph', ...
%                                  'data/processed/test', false);
%
% ============================================================

function [model, results] = run_comsol_case(field_path, template_path, output_dir, save_model)

if nargin < 4, save_model = false; end

addpath('C:\Program Files\COMSOL63\mli');
import com.comsol.model.*
import com.comsol.model.util.*

t_start = tic;
[~, name, ~] = fileparts(field_path);
if ~exist(output_dir,'dir'), mkdir(output_dir); end

case_path  = fullfile(output_dir, [name '.mph']);
export_csv = fullfile(output_dir, [name '_sol.csv']);

% --- Load template & assign interpolation -------------------------------
copyfile(template_path, case_path);
model = mphload(case_path);

model.func('int1').set('funcname','k');
model.func('int1').set('source','file');
model.func('int1').set('filename', field_path);
model.func('int1').set('nargs', 2);
model.func('int1').setIndex('argunit','m',0);
model.func('int1').setIndex('argunit','m',1);
model.func('int1').set('fununit','m^2');
model.func('int1').set('interp','linear');
model.func('int1').set('extrap','const');
model.func('int1').importData;

% --- Solve --------------------------------------------------------------
mphrun(model);   % <-- keine study1-Annahme mehr

% --- Export full field data ---------------------------------------------
if any(strcmp(model.result.export.tags,'data1'))
    expObj = model.result.export('data1');  % existiert bereits → wiederverwenden
else
    expObj = model.result.export.create('data1','Data');
end

expObj.set('data','dset1');
expObj.set('filename', export_csv);
expObj.set('expr',{ ...
    'x','y', ...
    'br.kappaxx','br.kappayx','br.kappazx', ...
    'br.kappaxy','br.kappayy','br.kappazy', ...
    'br.kappaxz','br.kappayz','br.kappazz', ...
    'br.U','u','v','p'});
expObj.set('header','on');
expObj.set('unit','on');
expObj.run;

% --- Optional: Save .mph file -------------------------------------------
if save_model
    mphsave(model, fullfile(output_dir, [name '_done.mph']));
end

ModelUtil.remove(model.tag);
delete(case_path);

% --- Metadata -----------------------------------------------------------
results = struct( ...
    'name', name, ...
    'field_path', field_path, ...
    'export_csv', export_csv, ...
    'save_model', save_model, ...
    'time_s', toc(t_start), ...
    'status', 'ok');
end
