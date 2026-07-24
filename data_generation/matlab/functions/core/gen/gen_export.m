% ============================================================
% Export 2D material and boundary-condition fields for COMSOL
%
% CSV columns (row-wise, flattened):
%   x ; y ;
%   Kxx ; Kxy ; Kyy ;
%   eps ;
%   [p_bc]
%
% JSON metadata:
%   - export schema
%   - present fields
%   - generator metadata
%
% Author: Rino M. Albertin
% Date:   2026-01-04
% ============================================================

function info_export = gen_export(fields, info, opts)

arguments
    fields struct
    info   struct
    opts   struct
end

if ~isfield(opts,'save_dir')
    error('gen_export_fields:MissingSaveDir', ...
        'opts.save_dir must be provided.');
end
if ~isfield(opts,'file_tag')
    opts.file_tag = "";
end
if ~isfield(opts,'delimiter')
    opts.delimiter = ';';
end

%% === Extract grid ==========================================
X = fields.grid.X;
Y = fields.grid.Y;

ny = size(X,1);
nx = size(X,2);

%% === Extract material fields ================================
Kxx = fields.material.K.Kxx;
Kyy = fields.material.K.Kyy;
Kxy = fields.material.K.Kxy;
eps = fields.material.eps;

%% === Boundary-condition field ===============================
p_bc = zeros(ny, nx);
p_bc(1,:) = fields.bc.p_inlet;

%% === Assemble CSV matrix ====================================
M = [
    X(:), ...
    Y(:), ...
    Kxx(:), ...
    Kxy(:), ...
    Kyy(:), ...
    eps(:), ...
    p_bc(:)
];

col_names = {'x','y','Kxx','Kxy','Kyy','eps','p_bc'};

%% === File naming ============================================
if strlength(opts.file_tag) > 0
    fname = char(opts.file_tag);
else
    fname = "fields_comsol";
end

if ~exist(opts.save_dir,'dir')
    mkdir(opts.save_dir);
end

path_csv  = fullfile(opts.save_dir, fname + ".csv");
path_json = fullfile(opts.save_dir, fname + ".json");

%% === Write CSV ==============================================
assert(numel(col_names) == size(M,2), ...
    'gen_export_fields:ColumnMismatch', ...
    'Number of column names does not match CSV matrix.');
writematrix(M, path_csv, 'Delimiter', opts.delimiter);

%% === JSON metadata ==========================================
info_export = struct();

info_export.export = struct( ...
    'file_base', fname, ...
    'delimiter', opts.delimiter, ...
    'columns',   {col_names} ...
);

info_export.fields_present = struct( ...
    'tensor',      true, ...
    'porosity',    true, ...
    'pressure_bc', true ...
);

generator = info;
if isfield(generator, 'geometry')
    generator = rmfield(generator, 'geometry');
end

info_export.geometry  = info.geometry;
info_export.generator = generator;

info_export.paths = struct( ...
    'csv',  path_csv, ...
    'json', path_json ...
);

info_export.timestamp = datestr(now,'yyyy-mm-dd HH:MM:SS');

fid = fopen(path_json,'w');
fprintf(fid,'%s', jsonencode(info_export,'PrettyPrint',true));
fclose(fid);

end
