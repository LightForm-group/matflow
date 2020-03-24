"""`matflow.software.matlab_mtex.py

Interface to `damask-parse`.

TODO: refactor get_load_case_* functions.

"""
from textwrap import dedent
from pathlib import Path

import numpy as np
from damask_parse import (read_geom, read_table,
                          write_geom, write_load_case, write_material_config)
from damask_parse.utils import get_header

from matflow import (
    TASK_INPUT_MAP,
    TASK_OUTPUT_MAP,
    TASK_FUNC_MAP,
    COMMAND_LINE_ARG_MAP,
    TASK_OUTPUT_FILES_MAP
)


def read_seeds_from_random(path):
    'Parse the file from the `seeds_fromRandom` DAMASK command.'

    header_lns = get_header(path)
    num_header_lns = len(header_lns)

    grid_size = None
    random_seed = None
    for ln in header_lns:
        if ln.startswith('grid'):
            grid_size = [int(j) for j in [i for i in ln.split()][1:][1::2]]
        if ln.startswith('randomSeed'):
            random_seed = int(ln.split()[1])

    data = np.loadtxt(path, skiprows=(num_header_lns + 1), ndmin=2)
    position = data[:, 0:3]
    eulers = data[:, 3:6]

    out = {
        'position': position,
        'eulers': eulers,
        'grid_size': grid_size,
        'random_seed': random_seed,
    }

    return out


def write_microstructure_seeds(path, microstructure_seeds):

    grid_size = microstructure_seeds['grid_size']
    position = microstructure_seeds['position']
    eulers = microstructure_seeds['eulers']

    data = np.hstack([position, eulers, np.arange(1, len(position) + 1)[:, None]])

    header = f"""
        3 header
        grid a {grid_size[0]} b {grid_size[1]} c {grid_size[2]}
        microstructures {len(data)}
        1_pos 2_pos 3_pos 1_euler 2_euler 3_euler microstructure
    """

    fmt = ['%20.16f'] * 6 + ['%10g']
    header = dedent(header).strip()
    np.savetxt(path, data, header=header, comments='', fmt=fmt)


def write_microstructure_new_orientations(path, microstructure_seeds, orientations):

    grid_size = microstructure_seeds['grid_size']
    position = microstructure_seeds['position']
    eulers = orientations['euler_angles']

    data = np.hstack([position, eulers, np.arange(1, len(position) + 1)[:, None]])

    header = f"""
        3 header
        grid a {grid_size[0]} b {grid_size[1]} c {grid_size[2]}
        microstructures {len(data)}
        1_pos 2_pos 3_pos 1_euler 2_euler 3_euler microstructure
    """

    fmt = ['%20.16f'] * 6 + ['%10g']
    header = dedent(header).strip()
    np.savetxt(path, data, header=header, comments='', fmt=fmt)


def get_load_case_random_2d(total_time, num_increments, normal_direction,
                            target_strain_rate=None, target_strain=None):

    if target_strain is None:
        target_strain = [None] * len(total_time)
    elif target_strain_rate is None:
        target_strain_rate = [None] * len(total_time)

    all_load_cases = []
    def_grad_vals = (np.random.random(4) - 0.5)
    for i, j, k, m, d in zip(total_time, num_increments, target_strain_rate,
                             target_strain, normal_direction):

        # Validation:
        msg = 'Specify either `target_strain_rate` or `target_strain`.'
        if all([t is None for t in [k, m]]):
            raise ValueError(msg)
        if all([t is not None for t in [k, m]]):
            raise ValueError(msg)

        dg_target_val = k or m
        if m:
            def_grad_vals *= dg_target_val
            def_grad_vals += np.eye(2).reshape(-1)
        elif k:
            def_grad_vals *= dg_target_val

        if d == 'x':
            # Deformation in the y-z plane

            dg_arr = np.ma.masked_array(
                [
                    [0, 0, 0],
                    [0, def_grad_vals[0], def_grad_vals[1]],
                    [0, def_grad_vals[2], def_grad_vals[3]],
                ],
                mask=np.array([
                    [1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0],
                ])
            )
            stress = np.ma.masked_array(
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                mask=np.array([
                    [0, 1, 1],
                    [0, 1, 1],
                    [0, 1, 1],
                ])
            )

        elif d == 'y':
            # Deformation in the x-z plane

            dg_arr = np.ma.masked_array(
                [
                    [def_grad_vals[0], 0, def_grad_vals[1]],
                    [0, 0, 0],
                    [def_grad_vals[2], 0, def_grad_vals[3]],
                ],
                mask=np.array([
                    [0, 1, 0],
                    [0, 1, 0],
                    [0, 1, 0],
                ])
            )
            stress = np.ma.masked_array(
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                mask=np.array([
                    [1, 0, 1],
                    [1, 0, 1],
                    [1, 0, 1],
                ])
            )

        elif d == 'z':
            # Deformation in the x-y plane

            dg_arr = np.ma.masked_array(
                [
                    [def_grad_vals[0], def_grad_vals[1], 0],
                    [def_grad_vals[2], def_grad_vals[3], 0],
                    [0, 0, 0],
                ],
                mask=np.array([
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 1],
                ])
            )
            stress = np.ma.masked_array(
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                mask=np.array([
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 0],
                ])
            )

        def_grad_aim = dg_arr if m else None
        def_grad_rate = dg_arr if k else None

        load_case = {
            'total_time': i,
            'num_increments': j,
            'def_grad_rate': def_grad_rate,
            'def_grad_aim': def_grad_aim,
            'stress': stress,
        }
        all_load_cases.append(load_case)

    return all_load_cases


def get_load_case_uniaxial(total_time, num_increments, direction, target_strain_rate=None,
                           target_strain=None):

    if target_strain is None:
        target_strain = [None] * len(total_time)
    elif target_strain_rate is None:
        target_strain_rate = [None] * len(total_time)

    # TODO: validate equal lengths of args (or `None`s)

    all_load_cases = []
    for i, j, k, m, d in zip(total_time, num_increments, target_strain_rate,
                             target_strain, direction):

        # Validation:
        msg = 'Specify either `target_strain_rate` or `target_strain`.'
        if all([t is None for t in [k, m]]):
            raise ValueError(msg)
        if all([t is not None for t in [k, m]]):
            raise ValueError(msg)

        dg_uniaxial_val = k or m

        # TODO: refactor:
        if d == 'x':
            dg_arr = np.ma.masked_array(
                [
                    [dg_uniaxial_val, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]
                ],
                mask=np.array([
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ])
            )

            stress = np.ma.masked_array(
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                mask=np.array([
                    [1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 0],
                ])
            )
        elif d == 'y':
            dg_arr = np.ma.masked_array(
                [
                    [0, 0, 0],
                    [0, dg_uniaxial_val, 0],
                    [0, 0, 0]
                ],
                mask=np.array([
                    [1, 0, 0],
                    [0, 0, 0],
                    [0, 0, 1],
                ])
            )

            stress = np.ma.masked_array(
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                mask=np.array([
                    [0, 1, 1],
                    [1, 1, 1],
                    [1, 1, 0],
                ])
            )
        elif d == 'z':
            dg_arr = np.ma.masked_array(
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, dg_uniaxial_val]
                ],
                mask=np.array([
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0],
                ])
            )

            stress = np.ma.masked_array(
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                mask=np.array([
                    [0, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1],
                ])
            )

        def_grad_aim = dg_arr if m else None
        def_grad_rate = dg_arr if k else None

        load_case = {
            'total_time': i,
            'num_increments': j,
            'def_grad_rate': def_grad_rate,
            'def_grad_aim': def_grad_aim,
            'stress': stress,
        }
        all_load_cases.append(load_case)

    return all_load_cases


def get_load_case_biaxial(total_time, num_increments, direction, target_strain_rate=None,
                          target_strain=None):

    if target_strain is None:
        target_strain = [None] * len(total_time)
    elif target_strain_rate is None:
        target_strain_rate = [None] * len(total_time)

    # TODO: validate equal lengths of args (or `None`s)

    all_load_cases = []
    for i, j, k, m, d in zip(total_time, num_increments, target_strain_rate,
                             target_strain, direction):

        # Validation:
        msg = 'Specify either `target_strain_rate` or `target_strain`.'
        if all([t is None for t in [k, m]]):
            raise ValueError(msg)
        if all([t is not None for t in [k, m]]):
            raise ValueError(msg)

        dg_biaxial_vals = k or m

        # TODO: refactor:
        if d == 'xy':
            dg_arr = np.ma.masked_array(
                [
                    [dg_biaxial_vals[0], 0, 0],
                    [0, dg_biaxial_vals[1], 0],
                    [0, 0, 0]
                ],
                mask=np.array([
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 1],
                ])
            )

            stress = np.ma.masked_array(
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                mask=np.array([
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 0],
                ])
            )
        elif d == 'xz':
            dg_arr = np.ma.masked_array(
                [
                    [dg_biaxial_vals[0], 0, 0],
                    [0, 0, 0],
                    [0, 0, dg_biaxial_vals[1]]
                ],
                mask=np.array([
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0],
                ])
            )

            stress = np.ma.masked_array(
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                mask=np.array([
                    [1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1],
                ])
            )
        elif d == 'yz':
            dg_arr = np.ma.masked_array(
                [
                    [0, 0, 0],
                    [0, dg_biaxial_vals[0], 0],
                    [0, 0, dg_biaxial_vals[1]]
                ],
                mask=np.array([
                    [1, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ])
            )

            stress = np.ma.masked_array(
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                mask=np.array([
                    [0, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                ])
            )

        def_grad_aim = dg_arr if m else None
        def_grad_rate = dg_arr if k else None

        load_case = {
            'total_time': i,
            'num_increments': j,
            'def_grad_rate': def_grad_rate,
            'def_grad_aim': def_grad_aim,
            'stress': stress,
        }
        all_load_cases.append(load_case)

    return all_load_cases


def get_load_case_plane_strain(total_time, num_increments, direction, target_strain_rate=None,
                               target_strain=None):

    if target_strain is None:
        target_strain = [None] * len(total_time)
    elif target_strain_rate is None:
        target_strain_rate = [None] * len(total_time)

    # TODO: validate equal lengths of args (or `None`s)

    all_load_cases = []
    for i, j, k, m, d in zip(total_time, num_increments, target_strain_rate,
                             target_strain, direction):

        # Validation:
        msg = 'Specify either `target_strain_rate` or `target_strain`.'
        if all([t is None for t in [k, m]]):
            raise ValueError(msg)
        if all([t is not None for t in [k, m]]):
            raise ValueError(msg)

        dg_ps_val = k or m

        # TODO: refactor:
        if d == 'xy':
            dg_arr = np.ma.masked_array(
                [
                    [dg_ps_val, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]
                ],
                mask=np.array([
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 1],
                ])
            )

            stress = np.ma.masked_array(
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                mask=np.array([
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 0],
                ])
            )
        elif d == 'zy':
            dg_arr = np.ma.masked_array(
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, dg_ps_val]
                ],
                mask=np.array([
                    [1, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ])
            )

            stress = np.ma.masked_array(
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                mask=np.array([
                    [0, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                ])
            )
        else:
            raise NotImplementedError()

        def_grad_aim = dg_arr if m else None
        def_grad_rate = dg_arr if k else None

        load_case = {
            'total_time': i,
            'num_increments': j,
            'def_grad_rate': def_grad_rate,
            'def_grad_aim': def_grad_aim,
            'stress': stress,
        }
        all_load_cases.append(load_case)

    return all_load_cases


def write_damask_load_case(path, load_case):
    write_load_case(path, load_case)


def write_damask_geom(path, volume_element):
    write_geom(volume_element, path)


def write_damask_material(path, material_properties, volume_element):
    write_material_config(material_properties, Path(path).parent, volume_element)


def read_damask_table(path):
    table_dat = read_table(path)
    return table_dat


def fmt_size(size):
    # TODO: # Some of these functions could also be imported from a "theme" package?
    return ' '.join(['{}'.format(i) for i in size])


COMMAND_LINE_ARG_MAP.update({
    ('generate_volume_element', 'random_voronoi_from_orientations', 'damask'): {
        'size': fmt_size,
    }
})

TASK_INPUT_MAP.update({
    ('generate_volume_element', 'random_voronoi', 'damask'): {
        'orientation.seeds': write_microstructure_seeds,
    },
    ('generate_volume_element', 'random_voronoi_from_orientations', 'damask'): {
        'orientation.seeds': write_microstructure_new_orientations,
    },
    ('simulate_volume_element_loading', 'CP_FFT', 'damask'): {
        'load.load': write_damask_load_case,
        'geom.geom': write_damask_geom,
        'material.config': write_damask_material,
    },
    ('visualise_volume_element', 'VTK', 'damask'): {
        'geom.geom': write_damask_geom,
    },
})

TASK_OUTPUT_MAP.update({
    ('generate_volume_element', 'random_voronoi', 'damask'): {
        'volume_element': read_geom,
    },
    ('generate_volume_element', 'random_voronoi_from_orientations', 'damask'): {
        'volume_element': read_geom,
    },
    ('generate_microstructure_seeds', 'random', 'damask'): {
        'microstructure_seeds': read_seeds_from_random,
    },
    ('simulate_volume_element_loading', 'CP_FFT', 'damask'): {
        'volume_element_response': read_damask_table,
    }
})

TASK_OUTPUT_FILES_MAP.update({
    ('visualise_volume_element', 'VTK', 'damask'): {
        '__file__VTR_file': 'geom.vtr',
    },
})

TASK_FUNC_MAP.update({
    ('generate_load_case', 'uniaxial'): get_load_case_uniaxial,
    ('generate_load_case', 'biaxial'): get_load_case_biaxial,
    ('generate_load_case', 'plane_strain'): get_load_case_plane_strain,
    ('generate_load_case', 'random_2d'): get_load_case_random_2d,
})
