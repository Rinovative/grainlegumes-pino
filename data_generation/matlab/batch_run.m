%% batch_run.m
% ============================================================
% Full batch pipeline: parameter sampling → κ(x,y) generation → COMSOL simulation
% Author: Rino M. Albertin
% Date: 2025-10-16
%
% DESCRIPTION
%   Executes a full batch of synthetic permeability fields:
%     1. Generates parameter samples (via sample_parameters.m)
%     2. Generates κ(x,y) fields using gen_permeability.m
%     3. Runs COMSOL Darcy–Brinkman simulations via run_comsol_case.m
%
%   Results are stored in:
%       data/raw/<batch_name>/       → generated κ(x,y)
%       data/processed/<batch_name>/ → COMSOL results
%
% REQUIREMENTS
%   • COMSOL Multiphysics with LiveLink for MATLAB
%   • Functions:
%       - sample_parameters.m
%       - gen_permeability.m
%       - run_comsol_case.m
% ============================================================

clear; clc;

%% --- Settings ----------------------------------------------------------
debug = true;             % 🧩 enable/disable debug output globally
dbstop if error           % optional: stop automatically on any error

%% --- Configuration -----------------------------------------------------
this_file  = mfilename('fullpath');
script_dir = fileparts(this_file);
project_root = fullfile(script_dir, '..');
project_root = char(java.io.File(project_root).getCanonicalPath());
addpath(genpath(fullfile(project_root, 'matlab', 'functions')));

% === SAMPLING PARAMETERS ===
method     = 'uniform';     % 'uniform', 'lhs', or 'sobol'
variation  = 0.20;          % ± % variation
N          = 1000;          % number of fields
seed       = 1;             % reproducibility
p_log      = 1.0;           % fraction of lognormal fields

batch_name = sprintf('samples_%s_var%.0f_N%d', method, variation*100, N);

% === PATHS ===
meta_dir      = fullfile(project_root, 'data', 'meta');
raw_dir       = fullfile(project_root, 'data', 'raw', batch_name);
processed_dir = fullfile(project_root, 'data', 'processed', batch_name);
template_path = fullfile(project_root, 'comsol', 'template_brinkman.mph');

if ~isfolder(meta_dir), mkdir(meta_dir); end
if ~isfolder(raw_dir), mkdir(raw_dir); end
if ~isfolder(processed_dir), mkdir(processed_dir); end

%% --- Generate or load parameter samples -------------------------------
sample_csv = fullfile(meta_dir, [batch_name '.csv']);

if ~isfile(sample_csv)
    disp("🧩 Kein Sample vorhanden – generiere neue Parameter:");
    sample_parameters(method, variation, N, seed, p_log, meta_dir);
else
    disp("📂 Verwende vorhandenes Sample:");
end

%% --- COMSOL connection -------------------------------------------------
addpath('C:\Program Files\COMSOL63\mli');
try
    v = mphversion;
    disp("✅ Verbunden mit COMSOL Server: " + v);
catch
    disp("🔄 Starte Verbindung zum COMSOL Server (Port 2036)...");
    mphstart(2036);
    pause(2);
    v = mphversion;
    disp("✅ Verbunden mit COMSOL Server: " + v);
end

%% --- Load sample parameters --------------------------------------------
T = readtable(sample_csv, 'Delimiter',';');
n_cases = height(T);

disp("------------------------------------------------------------");
disp("🚀 Starte Batch mit " + n_cases + " Fällen (" + batch_name + ")");
disp("------------------------------------------------------------");

%% --- Fixed geometry parameters ----------------------------------------
Lx = 1.2; 
Ly = 0.75; 
res = 0.003;

%% --- Start total timer -------------------------------------------------
t_batch_start = tic;

%% --- Main batch loop ---------------------------------------------------
for i = 1:n_cases
    case_id = T.case_id(i);

    opts = struct( ...
        'anisotropy', [T.ani_x(i), T.ani_y(i)], ...
        'volume_fraction', T.vol_frac(i), ...
        'ms_weight', [T.msW_c(i), T.msW_f(i)], ...
        'ms_scale', T.ms_scale(i), ...
        'coupling', T.coupling(i), ...
        'lognormal', logical(T.lognormal(i)), ...
        'save', true, ...
        'save_dir', raw_dir, ...
        'file_tag', char(sprintf('case_%04d', case_id)) ...
    );

    k_mean       = T.k_mean(i);
    var_rel      = T.var_rel(i);
    corr_len_rel = T.corr_len_rel(i);
    seed_case    = i; % unique seed per case

    %% --- Debug information --------------------------------------------
    if debug
        fprintf('\n[DEBUG] Case %d/%d\n', i, n_cases);
        fprintf('  k_mean=%.2e | var_rel=%.2f | corr_len_rel=%.3f | seed=%d\n', ...
            k_mean, var_rel, corr_len_rel, seed_case);
        fprintf('  anisotropy=[%.2f, %.2f] | vol_frac=%.2f | ms_scale=%.3f | coupling=%.2f\n', ...
            opts.anisotropy(1), opts.anisotropy(2), opts.volume_fraction, opts.ms_scale, opts.coupling);
        fprintf('  save_dir: %s\n', opts.save_dir);
        fprintf('  file_tag: %s\n', opts.file_tag);
    end

    %% --- Step 1: Generate field ---------------------------------------
    try
        [kappa, X, Y, info_field] = gen_permeability( ...
            Lx, Ly, res, k_mean, var_rel, corr_len_rel, seed_case, opts);
        if debug
            fprintf('  → κ-field generated: %s\n', info_field.file.path_csv);
        end
    catch ME
        fprintf('[%4d/%4d] ❌ Fehler bei gen_permeability: %s\n', i, n_cases, ME.message);
        if debug
            fprintf('   save_dir=%s | file_tag=%s\n', opts.save_dir, opts.file_tag);
        end
        continue;
    end

    %% --- Step 2: Run COMSOL simulation -------------------------------
    field_path = info_field.file.path_csv;
    save_model = false;

    try
        [model, results] = run_comsol_case(field_path, template_path, processed_dir, save_model);
        fprintf('[%4d/%4d] ✅ COMSOL fertig: %s (%.1f s)\n', ...
            i, n_cases, opts.file_tag, results.time_s);
    catch ME
        fprintf('[%4d/%4d] ❌ Fehler bei COMSOL: %s\n', i, n_cases, ME.message);
        if debug
            fprintf('   → field_path war: %s\n', field_path);
        end
        continue;
    end
end

%% --- End total timer ---------------------------------------------------
t_batch_end = toc(t_batch_start);
t_min = t_batch_end / 60;
t_hr  = t_batch_end / 3600;

disp("------------------------------------------------------------");
fprintf("🏁 Batch vollständig abgeschlossen.\n");
fprintf("⏱️  Gesamtzeit: %.1f s (%.2f min | %.2f h)\n", ...
    t_batch_end, t_min, t_hr);
disp("------------------------------------------------------------");

if debug
    dbclear if error
end
