function shape_crosscheck(shapeDir, inputDir, outFile)
%SHAPE_CROSSCHECK Analyse the meshes exported by export_crosscheck.py with the
% original MATLAB SHAPE (Angelidakis, Nadimi and Utili, CPC 265 (2021) 107983).
%
%   shape_crosscheck(shapeDir, inputDir, outFile)
%
%   shapeDir : folder "Matlab" of a SHAPE checkout (https://github.com/vsangelidakis/SHAPE)
%   inputDir : folder with the *.mat files written by export_crosscheck.py
%   outFile  : JSON file with the SHAPE results
%
% Each particle is built with SHAPE's Particle class (surface mesh input,
% tetrahedral volume via TetGen, alphaShape convex hull, PCA-based OBB of the
% surface points). The minimum-volume OBB of SHAPE ('minVolume' option) is
% evaluated separately with the same minboundbox call used by
% Auxiliary_geometries, so that its cost can be timed on its own.

addpath(genpath(fullfile(shapeDir, 'functions')));
addpath(genpath(fullfile(shapeDir, 'lib')));
addpath(genpath(fullfile(shapeDir, 'classes')));
loaded = load(fullfile(shapeDir, 'examples', 'options.mat'));
options = loaded.options;
options.warning = 'off';
options.meshcheckrepair = false;      % analyse the exact same surface as pySHAPE
options.useConvexHull = false;
options.Mesh.Voxelated_Image.div = 50;
options.Mesh.surf2mesh.maxvol = 1e9;  % volume is exact for any tetrahedralisation
options.Auxiliary_Geometries.OBB.method = 'PCA_points';
options.Auxiliary_Geometries.OBB.points = 'Surface_points';

files = dir(fullfile(inputDir, '*.mat'));
results = struct();
for k = 1:numel(files)
    [~, name] = fileparts(files(k).name);
    data = load(fullfile(inputDir, files(k).name));
    V = data.nodes; F = double(data.faces);

    started = tic;
    p = Particle(V, F, [], [], options);
    r.seconds_particle = toc(started);

    g = p.Original.Geometrical_features;
    form = p.Original.Morphological_features.Form;
    r.n_vertices = size(V, 1);
    r.n_faces = size(F, 1);
    r.volume = g.Volume;
    r.surface_area = g.Surface_area;
    r.centroid = g.Centroid;
    r.volume_convex_hull = g.Volume_CH;
    moments = sort(diag(g.Principal_inertia_tensor));
    r.inertia_I1 = moments(1); r.inertia_I2 = moments(2); r.inertia_I3 = moments(3);
    r.convexity = form.Convexity;
    r.sphericity_wadell = form.Sphericity_Wadell;
    r.orientation_C = form.Form_indices_sot.Compactness;
    r.orientation_F = form.Form_indices_sot.Flatness;
    r.orientation_R = form.Form_indices_sot.Elongation;
    r.pca = axis_indices(g.S_obb, g.I_obb, g.L_obb);

    % SHAPE's minVolume OBB (Auxiliary_geometries, minboundbox level 3)
    started = tic;
    [~, corners] = minboundbox(V(:, 1), V(:, 2), V(:, 3), 'v', 3);
    r.seconds_minboundbox = toc(started);
    D = [norm(corners(1, :) - corners(2, :)), norm(corners(1, :) - corners(4, :)), ...
         norm(corners(1, :) - corners(5, :))];
    D = sort(D);
    r.min_volume = axis_indices(D(1), D(2), D(3));

    % Cost of the two surface loops shared by both codes
    started = tic; surface_area(V, F); r.seconds_surface_area = toc(started);
    started = tic; SurfaceOrientationTensor(V, F); r.seconds_orientation_tensor = toc(started);

    results.(name) = r;
    fprintf('%-14s %6d faces  Particle %.2f s  minboundbox %.2f s\n', name, r.n_faces, ...
            r.seconds_particle, r.seconds_minboundbox);
    clear r
end

fid = fopen(outFile, 'w');
fprintf(fid, '%s', jsonencode(results, 'PrettyPrint', true));
fclose(fid);
end

function r = axis_indices(S, I, L)
[r.sphericity_krumbein, r.potticary_flatness, r.potticary_elongation, ...
 r.kongfonseca_flatness, r.kongfonseca_elongation, r.angelidakis_flatness, ...
 r.angelidakis_elongation, r.zingg_SI, r.zingg_IL] = Form_functions_2(S, I, L);
r.axis_S = S; r.axis_I = I; r.axis_L = L;
end
