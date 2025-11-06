%% visualize_case.m
% ============================================================
% Load and visualize 2D Darcy–Brinkman COMSOL results
% Author: Rino M. Albertin
% Date: 2025-10-14 (Updated: tab-compatible visualization)
%
% DESCRIPTION
%   Reads a COMSOL-exported .csv file containing field data of a
%   2D Darcy–Brinkman simulation and reconstructs the variables on
%   a regular grid for analysis or visualization.
%
%   Supported fields (depending on export structure):
%       - kxx, kyy  → permeability tensor components [m²]
%       - u, v, |U| → velocity components and magnitude [m/s]
%       - p         → pressure field [Pa]
%
%   The function automatically:
%       • Detects and skips header/comment lines (%)
%       • Reconstructs a regular mesh grid (x, y)
%       • Corrects COMSOL’s coordinate orientation
%       • Generates a 2×3 tiled plot layout:
%             log₁₀(kxx), log₁₀(kyy), p, |U|, v, u
%       • Computes simple field statistics for metadata
%
%   If a parent UI container (figure, uitab, or uipanel) is provided,
%   the plots are rendered inside it (e.g. one case per tab).
%   Otherwise, a new standalone figure window is created.
%
% INPUTS
%   file_path : string
%       Absolute or relative path to the COMSOL result .csv file.
%
%   parent    : (optional) graphics container handle
%       Handle to a figure, uitab, or uipanel where the plots
%       should be rendered. If omitted, a new figure is created.
%
% OUTPUTS
%   fields : struct
%       Contains 2D matrices of all reconstructed physical fields.
%         fields.kxx, fields.kyy, fields.u, fields.v, fields.Umag, fields.p
%
%   X, Y : double [ny × nx]
%       Regular mesh grid coordinates [m].
%
%   info : struct
%       Metadata including grid parameters, file path, and basic statistics.
%
% EXAMPLE
%   % Standalone visualization
%   visualize_case('data/processed/test_case_001_sol.csv');
%
%   % Visualization inside a tab
%   fig = figure; tg = uitabgroup(fig); t = uitab(tg, 'Title', 'Case 1');
%   visualize_case('case001_sol.csv', t);
%
% DEPENDENCIES
%   - MATLAB R2021b or later (for tiledlayout and uitabgroup)
%   - COMSOL-exported .csv file with standard column structure
% ============================================================

function [fields, X, Y, info] = visualize_case(file_path, parent)
%% --- Check file existence ----------------------------------------------
if ~isfile(file_path)
    error('File not found: %s', file_path);
end

%% --- Count header lines -------------------------------------------------
fid = fopen(file_path, 'r');
header_lines = 0;
while true
    tline = fgetl(fid);
    if ~ischar(tline), break; end
    if startsWith(strtrim(tline), '%')
        header_lines = header_lines + 1;
    else
        break;
    end
end
fclose(fid);
header_lines = header_lines + 1; % +1 for column header line

%% --- Import data --------------------------------------------------------
opts = detectImportOptions(file_path, ...
    'NumHeaderLines', header_lines, ...
    'VariableNamingRule', 'preserve');
T = readtable(file_path, opts);

%% --- Assign columns (fixed COMSOL export structure) ---------------------
x    = T.Var1;
y    = T.Var2;
kxx  = T.Var5;
kyy  = T.Var9;
Umag = T.Var14;
u    = T.Var15;
v    = T.Var16;
p    = T.Var17;

%% --- Clean data ---------------------------------------------------------
mask = ~(isnan(x) | isnan(y));
x = x(mask); y = y(mask);
kxx = kxx(mask); kyy = kyy(mask);
Umag = Umag(mask); u = u(mask); v = v(mask); p = p(mask);

[~, ia] = unique([x, y], 'rows', 'stable');
x = x(ia); y = y(ia);
kxx = kxx(ia); kyy = kyy(ia);
Umag = Umag(ia); u = u(ia); v = v(ia); p = p(ia);

%% --- Reconstruct regular grid ------------------------------------------
x_unique = unique(x);
y_unique = unique(y);
nx = numel(x_unique);
ny = numel(y_unique);

if abs(numel(x) - nx*ny) <= (nx*ny*0.1)
    [Xg, Yg] = meshgrid(x_unique, y_unique);
    X = Xg; Y = Yg;
    Fkxx  = scatteredInterpolant(x, y, kxx);
    Fkyy  = scatteredInterpolant(x, y, kyy);
    FUmag = scatteredInterpolant(x, y, Umag);
    Fu    = scatteredInterpolant(x, y, u);
    Fv    = scatteredInterpolant(x, y, v);
    Fp    = scatteredInterpolant(x, y, p);
    fields.kxx  = Fkxx(X, Y);
    fields.kyy  = Fkyy(X, Y);
    fields.Umag = FUmag(X, Y);
    fields.u    = Fu(X, Y);
    fields.v    = Fv(X, Y);
    fields.p    = Fp(X, Y);
else
    X = reshape(x, ny, nx);
    Y = reshape(y, ny, nx);
    fields.kxx  = reshape(kxx, ny, nx);
    fields.kyy  = reshape(kyy, ny, nx);
    fields.Umag = reshape(Umag, ny, nx);
    fields.u    = reshape(u, ny, nx);
    fields.v    = reshape(v, ny, nx);
    fields.p    = reshape(p, ny, nx);
end

%% --- Correct orientation (COMSOL → MATLAB) ------------------------------
fields.kxx  = flipud(fields.kxx);
fields.kyy  = flipud(fields.kyy);
fields.Umag = flipud(fields.Umag);
fields.u    = flipud(fields.u);
fields.v    = flipud(fields.v);
fields.p    = flipud(fields.p);

%% --- Metadata -----------------------------------------------------------
info = struct();
info.file = file_path;
info.grid = struct('nx', nx, 'ny', ny, ...
                   'x_range', [min(x_unique), max(x_unique)], ...
                   'y_range', [min(y_unique), max(y_unique)]);

%% --- Visualization ------------------------------------------------------

% ✅ Create a valid drawing parent
if nargin < 2 || isempty(parent)
    fig = figure('Units','normalized','Position',[0.05 0.1 0.9 0.7]);
    parent = fig; % standalone mode
else
    % For uitab or uifigure support, embed plots into a panel
    parent = uipanel('Parent', parent, 'Units', 'normalized', ...
                     'Position', [0 0 1 1], 'BorderType', 'none');
end

tl = tiledlayout(parent, 2, 3, 'Padding', 'compact', 'TileSpacing', 'compact');

[~, fname, ~] = fileparts(file_path);
sgtitle(tl, strrep(fname, '_', '\_'), 'FontWeight', 'bold', 'FontSize', 14);

colormap(turbo(10));

titles = {'$\log_{10}(k_{xx})$', '$\log_{10}(k_{yy})$', 'Pressure [Pa]', ...
          '$|U|$ [m/s]', '$v$ [m/s]', '$u$ [m/s]'};
imgs = {log10(fields.kxx), log10(fields.kyy), fields.p, fields.Umag, fields.v, fields.u};

for i = 1:numel(imgs)
    ax = nexttile(tl);
    imagesc(ax, x_unique, y_unique, imgs{i});
    axis(ax, 'equal', 'tight');
    cb = colorbar(ax);
    cb.TickDirection = 'out';
    title(ax, titles{i}, 'Interpreter', 'latex', 'FontWeight', 'bold');
end
end