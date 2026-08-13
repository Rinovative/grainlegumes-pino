%% visualize_case.m
% ============================================================
% Load and visualize one Darcy-Brinkman COMSOL result.
%
% Author: Rino M. Albertin
% Date:   2026-08-13
%
% DESCRIPTION
%   Reads the exact semicolon-delimited export produced by run_comsol_case,
%   validates that it represents one finite and complete Cartesian grid,
%   reconstructs all fields with deterministic ascending axes, and renders a
%   two-by-three diagnostic plot.
%
% EXPORT COLUMN CONTRACT
%   The file must contain exactly these 12 numeric columns after any leading
%   lines whose first non-whitespace character is "%":
%
%       1  x          5  kappaxy      9  p
%       2  y          6  kappayy     10  u
%       3  kappaxx    7  eps         11  v
%       4  kappayx    8  p_bc        12  U
%
%   Coordinates are sorted by x and then y, checked for duplicates and missing
%   Cartesian points, and reshaped to [ny, nx]. The exported xy and yx tensor
%   entries are averaged to produce the reconstructed kxy field.
%
% VISUALIZATION
%   The six tiles show, in order:
%
%       log10(kxx), log10(kyy), pressure, speed, v velocity, u velocity.
%
%   Each tile uses physical x/y coordinates, equal axis scaling, a tight view,
%   and its own colorbar. With no parent, the function creates a standalone
%   figure. With a parent, it creates a borderless panel inside that graphics
%   container and places the tiled layout there.
%
% USAGE
%   [fields, X, Y, info] = visualize_case(file_path)
%   [fields, X, Y, info] = visualize_case(file_path, parent)
%
% INPUTS
%   file_path
%       Absolute or relative path to one run_comsol_case solution CSV.
%   parent
%       Optional figure, tab, or panel that receives the plot layout.
%
% OUTPUTS
%   fields
%       Reconstructed [ny, nx] arrays kxx, kxy, kyy, eps, p_bc, p, u, v,
%       and Umag.
%   X, Y
%       Cartesian mesh coordinates with ascending x columns and y rows.
%   info
%       Source file path, nx, ny, and x/y coordinate ranges.
%
% EXAMPLES
%   visualize_case('data/processed/test_case_001_sol.csv');
%
%   fig = figure;
%   tabs = uitabgroup(fig);
%   tab = uitab(tabs, 'Title', 'Case 1');
%   visualize_case('case001_sol.csv', tab);
%
% NOTES
%   The reader rejects non-finite values, wrong column counts, duplicate points,
%   incomplete grids, and coordinates that cannot be reconstructed exactly.
%   It consumes an existing export and does not require a live COMSOL model.
% ============================================================

function [fields, X, Y, info] = visualize_case(file_path, parent)

%% --- Check file existence ----------------------------------------------
if ~isfile(file_path)
    error('File not found: %s', file_path);
end

%% --- Read exact runner export contract ---------------------------------
fid = fopen(file_path, 'r');
if fid < 0
    error('visualize_case:OpenFile', 'Could not open file: %s', file_path);
end
file_cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
header_lines = 0;
while true
    tline = fgetl(fid);
    if ~ischar(tline) || ~startsWith(strtrim(tline), '%')
        break;
    end
    header_lines = header_lines + 1;
end
clear file_cleanup

data = readmatrix(file_path, 'Delimiter', ';', ...
    'NumHeaderLines', header_lines);
expected_columns = 12;
if isempty(data) || size(data, 2) ~= expected_columns
    error('visualize_case:ExportContract', ...
        ['Expected 12 columns from run_comsol_case: x, y, kappaxx, kappayx, ' ...
        'kappaxy, kappayy, eps, p_bc, p, u, v, and U.']);
end
if any(~isfinite(data), 'all')
    error('visualize_case:NonfiniteData', ...
        'COMSOL result contains non-finite values: %s', file_path);
end

%% --- Validate and reconstruct deterministic Cartesian grid -------------
data = sortrows(data, [1, 2]);
x = data(:, 1);
y = data(:, 2);
x_unique = unique(x, 'sorted');
y_unique = unique(y, 'sorted');
nx = numel(x_unique);
ny = numel(y_unique);
[X, Y] = meshgrid(x_unique, y_unique);
if size(unique([x, y], 'rows'), 1) ~= size(data, 1) || ...
        size(data, 1) ~= nx * ny || ...
        ~isequal([x, y], [X(:), Y(:)])
    error('visualize_case:CartesianGrid', ...
        'COMSOL coordinates must form one complete Cartesian grid without duplicates.');
end

kappa_yx = reshape(data(:, 4), ny, nx);
kappa_xy = reshape(data(:, 5), ny, nx);
fields = struct( ...
    'kxx', reshape(data(:, 3), ny, nx), ...
    'kxy', (kappa_yx + kappa_xy) / 2, ...
    'kyy', reshape(data(:, 6), ny, nx), ...
    'eps', reshape(data(:, 7), ny, nx), ...
    'p_bc', reshape(data(:, 8), ny, nx), ...
    'p', reshape(data(:, 9), ny, nx), ...
    'u', reshape(data(:, 10), ny, nx), ...
    'v', reshape(data(:, 11), ny, nx), ...
    'Umag', reshape(data(:, 12), ny, nx));

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