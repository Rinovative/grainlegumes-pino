%% visualize_case.m
% ============================================================
% Load and visualize 2D Darcy–Brinkman COMSOL results
% Author: Rino M. Albertin
% Date: 2025-10-14
%
% DESCRIPTION
%   Reads a COMSOL-exported .csv file containing field data of a
%   2D Darcy–Brinkman simulation and reconstructs the variables on
%   a regular grid for analysis or visualization.
%
%   Supported fields (the exact run_comsol_case export contract):
%       - kxx, kxy, kyy → permeability tensor components [m²]
%       - eps, p_bc     → porosity and inlet pressure fields
%       - p, u, v, |U|  → solved pressure and velocity fields
%
%   The function automatically:
%       • Detects and skips header/comment lines (%)
%       • Validates a complete 12-column Cartesian-grid export
%       • Reconstructs deterministic ascending x/y orientation
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
%         fields.kxx, fields.kxy, fields.kyy, fields.eps, fields.p_bc,
%         fields.p, fields.u, fields.v, fields.Umag
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