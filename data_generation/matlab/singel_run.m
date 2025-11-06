%% ============================================================
%  PIPELINE_RUN_SINGLE
%  Full pipeline: generate κ(x,y) → COMSOL simulation → visualize results
%  Author: Rino M. Albertin
%  Date: 2025-10-14
%
%  DESCRIPTION
%  -------------
%  1. Generates a synthetic permeability field κ(x,y)
%  2. Runs a Darcy–Brinkman COMSOL case using template_brinkman.mph
%  3. Loads and visualizes the COMSOL result fields
%
%  REQUIREMENTS
%  -------------
%  • COMSOL Multiphysics with LiveLink for MATLAB
%  • Functions:
%       - gen_permeability.m
%       - run_comsol_case.m
%       - visualize_case.m
% ============================================================

clear; clc; close all;

%% --- Paths --------------------------------------------------------------
this_file  = mfilename('fullpath');
script_dir = fileparts(this_file);
project_root = fullfile(script_dir, '..');
project_root = char(java.io.File(project_root).getCanonicalPath());

addpath(genpath(fullfile(project_root, 'matlab', 'functions')));

raw_dir      = fullfile(project_root, 'data', 'raw', 'pipeline_test');
processed_dir = fullfile(project_root, 'data', 'processed', 'pipeline_test');
template_path = fullfile(project_root, 'comsol', 'template_brinkman.mph');

if ~isfolder(raw_dir), mkdir(raw_dir); end
if ~isfolder(processed_dir), mkdir(processed_dir); end

%% --- COMSOL Connection --------------------------------------------------
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

%% --- 1️⃣ Generate permeability field -----------------------------------
disp("------------------------------------------------------------");
disp("🧩 Schritt 1: Generiere Permeabilitätsfeld κ(x,y)");
disp("------------------------------------------------------------");

Lx = 1.2; Ly = 0.75; res = 0.003;
k_mean = 5e-9; var_rel = 0.5; corr_len_rel = 0.05; seed = 1;

opts = struct( ...
    "lognormal", true, ...
    "anisotropy", [3,1], ...
    "ms_weight", [0.7,0.3], ...
    "ms_scale", 0.1, ...
    "volume_fraction", 1.0, ...
    "coupling", 0.0, ...
    "save", true, ...
    "save_dir", raw_dir ...
);

[kappa, X, Y, info_field] = gen_permeability( ...
    Lx, Ly, res, k_mean, var_rel, corr_len_rel, seed, opts);

disp("✅ Feld gespeichert unter:");
disp("   " + info_field.file.path_csv);

%% --- 2️⃣ Run COMSOL simulation ----------------------------------------
disp("------------------------------------------------------------");
disp("⚙️  Schritt 2: Starte COMSOL-Simulation");
disp("------------------------------------------------------------");

field_path = info_field.file.path_csv;
save_model = false;

[model, results] = run_comsol_case(field_path, template_path, processed_dir, save_model);

disp("✅ COMSOL abgeschlossen:");
disp("   Export: " + results.export_csv);
disp("   Dauer: " + sprintf('%.1f s', results.time_s));

%% --- 3️⃣ Visualize COMSOL results -------------------------------------
disp("------------------------------------------------------------");
disp("📊 Schritt 3: Visualisiere Resultate");
disp("------------------------------------------------------------");

[field_data, Xc, Yc, info_vis] = visualize_case(results.export_csv);

disp("🏁 Pipeline erfolgreich abgeschlossen!");
disp("------------------------------------------------------------");