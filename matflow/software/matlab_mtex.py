"""`matflow.software.matlab_mtex.py

Interface to the MTEX Matlab toolbox.

"""

from pathlib import Path
from textwrap import dedent

import numpy as np

from matflow import TASK_INPUT_MAP, TASK_OUTPUT_MAP


def prepare_script_estimate_ODF(path, CTF_file_path, rotation_angle):

    script_txt = f"""
        % Specify crystal symmetry
        crystalSymArray = {{... 
            'notIndexed', ...
            crystalSymmetry(...
                'm-3m',...
                [4.05 4.05 4.05],...
                'mineral', 'Aluminium',...
                'color', 'light blue'...
            )}};

        % Plotting convention
        setMTEXpref('xAxisDirection','east');
        setMTEXpref('zAxisDirection','outOfPlane');

        %% Import the Data
        ebsd = loadEBSD(...
            '{CTF_file_path}',...
            crystalSymArray,...
            'interface', 'ctf',...
            'convertEuler2SpatialReferenceFrame'...
        );

        rot = rotation('axis', xvector, 'angle', {rotation_angle}*degree);
        ebsd = rotate(ebsd, rot);

        ebsd = ebsd('indexed');
        [grains, ebsd.grainId] = calcGrains(ebsd,'angle', 5*degree);
        grains = smooth(grains);

        plot(ebsd,ebsd.bc);
        colormap gray
        mtexColorbar;

        hold on

        oM = ipfHSVKey(ebsd('indexed'))
        oM.inversePoleFigureDirection = xvector;
        color = oM.orientation2color(ebsd('indexed').orientations);
        plot(ebsd('indexed'), color);
        saveas(gcf,'EBSD.png')

        hold off

        figure()

        ori=ebsd('Aluminium').orientations
        x=[...
            Miller(1, 1, 1, ori.CS),...
            Miller(2,0,0,ori.CS),...
            Miller(2,2,0,ori.CS)...
        ];
        plotPDF(ori, x, 'antipodal', 'contourf', 'colorrange', [1 3.5])
        colorbar;
        saveas(gcf,'pole_figs.png')

        figure()

        ori = ebsd('Aluminium').orientations;
        ori.SS = specimenSymmetry('orthorhombic');
        odf = calcODF(ori);
        plot(odf,'phi2',[0 45 65]* degree,'antipodal','linewidth',2,'colorbar');
        saveas(gcf,'odf_slices.png')
        ori1 = calcOrientations(odf,2000);
        export(ori1, 'ori1.txt');

        export(odf, 'odf.txt', 'Bunge', 'MTEX');

    """

    script_txt = dedent(script_txt)
    with Path(path).open('w') as handle:
        handle.write(script_txt)


def parse_mtex_ODF_file(path):
    print('parse_mtex_ODF_file')

    crystal_sym = None
    specimen_sym = None
    euler_angle_labels = None

    with Path(path).open() as handle:
        for ln_idx, ln in enumerate(handle.readlines()):
            ln_s = ln.strip()
            if ln_s.startswith('%'):
                if 'crystal symmetry' in ln_s:
                    crystal_sym = ln_s.split('"')[1]
                if 'specimen symmetry' in ln_s:
                    specimen_sym = ln_s.split('"')[1]
            if ln_s.endswith('value'):
                euler_angle_labels = ln_s.split()[1:4]
            if crystal_sym and specimen_sym and euler_angle_labels:
                break

    data = np.loadtxt(str(path), skiprows=4)
    euler_angles = data[:, 0:3]
    weights = data[:, 3]

    ODF = {
        'crystal_symmetry': crystal_sym,
        'specimen_symmetry': specimen_sym,
        'euler_angle_labels': euler_angle_labels,
        'euler_angles': euler_angles,
        'weights': weights,
    }
    return ODF


def prepare_script_sample_texture(path, ODF, num_orientations):

    print('prepare_script_sample_texture')

    # Write out ODF into an "MTEX" text file:
    base_path = Path(path).parent
    odf_path = base_path.joinpath('odf.txt')
    ang_labs = ODF['euler_angle_labels']

    odf_header = dedent(f"""
        % MTEX ODF
        % crystal symmetry: "{ODF['crystal_symmetry']}"
        % specimen symmetry: "{ODF['specimen_symmetry']}"
        % {ang_labs[0]} {ang_labs[1]} {ang_labs[2]} value
    """).strip()

    data = np.hstack([ODF['euler_angles'], ODF['weights'][:, None]])
    np.savetxt(odf_path, data, header=odf_header, fmt='%8.5f', comments='')

    # Write script to load ODF text file and sample orientations:
    script_txt = f"""
        % Define crystal and specimen symmetry:
        cs = crystalSymmetry('cubic');
        ss = specimenSymmetry('orthorhombic');

        % Load the data:
        odf = loadODF(...
            '{str(odf_path.name)}',...
            'cs', cs,...
            'ss', ss,...
            'ColumnNames', {{'Euler 1' 'Euler 2' 'Euler 3' 'weight'}}...
        );    
        
        plot(odf, 'phi2', [0 45 65]* degree, 'antipodal', 'linewidth', 2, 'colorbar');
        saveas(gcf, 'odf_slices_2.png')

        ori = calcOrientations(odf, {num_orientations});
        export(ori, 'orientations.txt');
    """

    script_txt = dedent(script_txt)
    with Path(path).open('w') as handle:
        handle.write(script_txt)


def parse_orientations(path):
    print('parse_orientations')

    with Path(path).open() as handle:
        ln = handle.readline()
        euler_angle_labels = ln.split()

    euler_angles = np.loadtxt(str(path), skiprows=1)

    orientations = {
        'euler_angle_labels': euler_angle_labels,
        'euler_angles': euler_angles,
    }
    return orientations


def prepare_script_model_ODF_uniform(path, crystal_symmetry, specimen_symmetry):

    script_txt = f"""
        cs = crystalSymmetry('{crystal_symmetry}');
        ss = specimenSymmetry('{specimen_symmetry}');
        odf = uniformODF(cs, ss)        
        export(odf, 'odf.txt', 'Bunge', 'MTEX');
    """

    script_txt = dedent(script_txt).strip()
    with Path(path).open('w') as handle:
        handle.write(script_txt)


TASK_INPUT_MAP.update({
    ('model_ODF', 'uniform_ODF', 'matlab_mtex'): {
        'create_model_ODF.m': prepare_script_model_ODF_uniform,
    },
    ('estimate_ODF', 'from_CTF_file', 'matlab_mtex'): {
        'estimate_ODF_from_CTF.m': prepare_script_estimate_ODF,
    },
    ('sample_texture', 'from_ODF', 'matlab_mtex'): {
        'sample_texture_from_ODF.m': prepare_script_sample_texture,
    }
})

TASK_OUTPUT_MAP.update({
    ('model_ODF', 'uniform_ODF', 'matlab_mtex'): {
        'ODF': parse_mtex_ODF_file,
    },
    ('estimate_ODF', 'from_CTF_file', 'matlab_mtex'): {
        'ODF': parse_mtex_ODF_file,
    },
    ('sample_texture', 'from_ODF', 'matlab_mtex'): {
        'orientations': parse_orientations,
    }
})
