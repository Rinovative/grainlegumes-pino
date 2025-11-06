%% ============================================================
%  Batch generation of synthetic permeability fields κ(x,y)
%  (Multiscale, lognormal, anisotropic)
%  Author: Rino M. Albertin
%  Date: 2025-10-16
%
%  Generates multiple random realizations of permeability fields
%  using the updated gen_permeability.m (2025-10-16, continuous param model).
%
%  Each field differs only by its random seed and is saved as:
%       • .csv  → numeric κ(x,y)
%       • .json → metadata (geometry, parameters, statistics)
%
%  Output directory (relative to this file):
%       ../../data/raw/test/
%
%  Note:
%  The "anisotropy" parameter [ax, ay] defines geometric (statistical)
%  anisotropy in the correlation structure of κ(x,y), i.e. elongated
%  spatial patterns. The resulting fields remain physically isotropic
%  (scalar permeability, no directional tensor effects).
% ============================================================

clear; clc;

%% --- Global control switches ------------------------------------------
SHOW_PLOTS = true;   % <-- toggle field visualization ON/OFF

%% --- Setup paths -------------------------------------------------------
this_dir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(this_dir, '..')));

opts.save_dir = fullfile(this_dir, '..', '..', '..', 'data', 'raw', 'test');
opts.save_dir = char(java.io.File(opts.save_dir).getCanonicalPath());

%% --- Base parameters ---------------------------------------------------
Lx = 1.2; Ly = 0.75;       % domain size [m]
res = 0.003;               % isotropic grid spacing [m]
k_mean = 5e-9;             % mean permeability [m²]
var_rel = 0.5;             % relative variance
corr_len_rel = 0.05;       % correlation length (relative)
seed_list = 1:4;           % number of realizations

%% --- Field generation options -----------------------------------------
opts.lognormal       = true;         % ensures κ > 0
opts.anisotropy      = [3,1];        % horizontal elongation
opts.ms_weight       = [0.7, 0.3];   % 70% coarse / 30% fine
opts.ms_scale        = 0.1;          % fine-scale correlation length factor
opts.volume_fraction = 1.0;          % full field (no sparsity)
opts.coupling        = 0.5;          % no cross-scale correlation
opts.save            = true;         % enable saving

%% --- Batch generation --------------------------------------------------
fprintf('Generating %d synthetic κ-fields...\n\n', numel(seed_list));

for i = 1:numel(seed_list)
    seed = seed_list(i);
    fprintf('→ Generating field %d/%d (seed = %d)...\n', i, numel(seed_list), seed);

    [kappa, X, Y, info] = gen_permeability( ...
        Lx, Ly, res, k_mean, var_rel, corr_len_rel, seed, opts);

    fprintf('   Saved to: %s\n', info.file.path_csv);

    %% --- Optional visualization ---------------------------------------
    if SHOW_PLOTS
        figure('Name', sprintf('κ-field seed=%d', seed));

        % Left: linear scale
        subplot(1,2,1);
        imagesc(X(1,:), Y(:,1), kappa);
        set(gca, 'YDir', 'normal');
        axis equal tight;
        title(sprintf('Linear scale (seed = %d)', seed), ...
            'FontWeight', 'bold', 'Interpreter', 'latex');
        xlabel('$x$ [m]', 'Interpreter', 'latex');
        ylabel('$y$ [m]', 'Interpreter', 'latex');
        cb1 = colorbar;
        ylabel(cb1, '$\kappa$ [m$^2$]', 'Interpreter', 'latex');

        % Right: log10 scale
        subplot(1,2,2);
        imagesc(X(1,:), Y(:,1), log10(kappa));
        set(gca, 'YDir', 'normal');
        axis equal tight;
        title('$\log_{10}(\kappa)$ scale', 'FontWeight', 'bold', ...
            'Interpreter', 'latex');
        xlabel('$x$ [m]', 'Interpreter', 'latex');
        ylabel('$y$ [m]', 'Interpreter', 'latex');
        cb2 = colorbar;
        ylabel(cb2, '$\log_{10}(\kappa)$', 'Interpreter', 'latex');

        % 🔧 updated: info.parameters.method no longer exists
        sgtitle(sprintf('$\\textbf{Permeability field}$'), ...
            'Interpreter', 'latex', 'FontWeight', 'bold', 'FontSize', 12);

        colormap("turbo")
        drawnow;
    end
end

fprintf('\n✅ All %d fields successfully generated and saved.\n', numel(seed_list));