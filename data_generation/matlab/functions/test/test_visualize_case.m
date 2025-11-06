%% TEST_VISUALIZE_CASE – Lädt und visualisiert alle COMSOL-Ergebnisse
clear; clc; close all;
NR_CASES_SHOW = 5;

%% --- Projektstruktur ---
this_file  = mfilename('fullpath');
script_dir = fileparts(this_file);
project_root = fullfile(script_dir, '..', '..', '..');
project_root = char(java.io.File(project_root).getCanonicalPath());

output_dir = fullfile(project_root, 'data', 'processed', 'samples_uniform_var5_N100');
addpath(genpath(fullfile(project_root, 'matlab', 'functions')));

assert(isfolder(output_dir), "❌ Ergebnisordner fehlt: " + string(output_dir));

%% --- Alle COMSOL-Ergebnisse finden ---
file_list = dir(fullfile(output_dir, '*_sol.csv'));
n_cases = numel(file_list);
assert(n_cases > 0, "❌ Keine COMSOL-Resultate im Output-Ordner gefunden.");

disp("------------------------------------------------------------");
disp("📊 Lade und visualisiere " + n_cases + " Ergebnisdateien aus:");
disp("📁 " + output_dir);
disp("------------------------------------------------------------");

%% --- Hauptfenster mit Tabs erstellen ---
n_cases = min(n_cases, NR_CASES_SHOW);
fig = figure('Name','COMSOL Results (Tabbed)', ...
             'Units','normalized', ...
             'Position',[0.05 0.05 0.9 0.85]);
tg = uitabgroup(fig);

%% --- Alle Files laden und visualisieren ---
for i = 1:n_cases
    f = file_list(i);
    file_path = fullfile(f.folder, f.name);
    disp("▶ [" + i + "/" + n_cases + "] " + f.name);

    try
        % Tab erzeugen
        [~, name, ~] = fileparts(f.name);
        tab = uitab(tg, 'Title', sprintf('%02d – %s', i, name));

        % visualize_case in diesen Tab zeichnen lassen
        [fields, X, Y, info] = visualize_case(file_path, tab);

    catch ME
        disp("   ❌ Fehler beim Einlesen oder Plotten: " + ME.message);
    end

    disp("------------------------------------------------------------");
end

disp("🏁 Alle Resultate erfolgreich visualisiert.");
disp("------------------------------------------------------------");
