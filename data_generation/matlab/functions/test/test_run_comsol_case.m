%% TEST_RUN_BATCH – Führt alle COMSOL-Cases im Test-Ordner aus
clear; clc;

%% --- COMSOL LiveLink Pfad hinzufügen ---
addpath('C:\Program Files\COMSOL64\mli');

% Verbindung prüfen / starten
try
    v = mphversion;
    disp("✅ Verbunden mit COMSOL Server: " + v);
catch
    disp('🔄 Starte Verbindung zum COMSOL Server (Port 2036)...');
    mphstart(2036);
    pause(2);
    v = mphversion;
    disp("✅ Verbunden mit COMSOL Server: " + v);
end

%% --- Projektstruktur (robust, relativ zum Speicherort dieses Skripts) ---
this_file  = mfilename('fullpath');
script_dir = fileparts(this_file);
project_root = fullfile(script_dir, '..', '..', '..');
project_root = char(java.io.File(project_root).getCanonicalPath());

raw_dir   = fullfile(project_root, 'data', 'raw', 'test');
template_path = fullfile(project_root, 'comsol', 'template_brinkman.mph');
output_dir    = fullfile(project_root, 'data', 'processed', 'test');
addpath(genpath(fullfile(project_root, 'matlab', 'functions')));

%% --- Existenz prüfen ---
assert(isfolder(raw_dir),    "❌ Eingabeordner fehlt: " + string(raw_dir));
assert(isfile(template_path),"❌ Template fehlt: " + string(template_path));
if ~isfolder(output_dir), mkdir(output_dir); end

%% --- Laufparameter ---
save_model = false; % true = .mph speichern
file_list = dir(fullfile(raw_dir, '*.csv'));
n_cases = numel(file_list);
assert(n_cases > 0, "❌ Keine CSV-Dateien im Eingabeordner gefunden.");

disp("------------------------------------------------------------");
disp("🚀 Starte Batchlauf mit " + n_cases + " Fällen:");
disp("Template : " + string(template_path));
disp("Output   : " + string(output_dir));
disp("Speichern: " + string(save_model));
disp("------------------------------------------------------------");

%% --- Batchlauf ---
for i = 1:n_cases
    f = file_list(i);
    field_path = fullfile(f.folder, f.name);
    case_name = erase(f.name, '.csv');

    disp("▶ [" + i + "/" + n_cases + "] " + case_name);

    try
        results = run_comsol_case(field_path, template_path, output_dir, save_model);
        assert(isnumeric(results.comsol_solve_s) && ...
            isscalar(results.comsol_solve_s) && ...
            isfinite(results.comsol_solve_s) && ...
            results.comsol_solve_s > 0, ...
            'COMSOL solve timing must be a finite positive scalar.');
        assert(results.time_s >= results.comsol_solve_s, ...
            'Complete COMSOL case time cannot be below solve time.');
        disp("   ✅ Erfolgreich (" + sprintf('%.1f', results.time_s) + " s)");
        disp("   → Solver: " + sprintf('%.6f', ...
            results.comsol_solve_s) + " s");
        disp("   → Export: " + results.export_csv);
        if results.save_model
            disp("   → Model saved (.mph)");
        end
    catch ME
        disp("   ❌ Fehler: " + ME.message);
    end

    disp("------------------------------------------------------------");
end

disp("🏁 Alle Fälle abgeschlossen.");
disp("------------------------------------------------------------");
