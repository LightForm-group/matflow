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
from vecmaths.rotation import get_random_rotation_matrix

from matflow import TASK_INPUT_MAP, TASK_OUTPUT_MAP, TASK_FUNC_MAP


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


def get_load_case_random_3d(total_time, num_increments, target_strain):

    def defgradScale(defgrad, finalStrain):

        def fill_star(a, b):
            if a != '*' and b != '*':
                return a, b
            elif a == '*' and b != '*':
                return b, b
            elif a != '*' and b == '*':
                return a, a
            else:
                return 0.0, 0.0

        defgrad0 = defgrad[:]  # AP: copy defgrad

        # Remove star elements (i.e. undefined) from off-diagonals;
        # this is just for calculating the factor
        # by which to scale defgrad to generate the specified equivalent strain.
        defgrad0[1], defgrad0[3] = fill_star(defgrad[1], defgrad[3])
        defgrad0[2], defgrad0[6] = fill_star(defgrad[2], defgrad[6])
        defgrad0[5], defgrad0[7] = fill_star(defgrad[5], defgrad[7])

        for i in [0, 4, 8]:
            if defgrad0[i] == '*':
                defgrad0[i] = 0.0

        # AP: note that the determinant gives the volume scaling factor of a linear
        # transformation...
        det0 = 1.0 - np.linalg.det(np.array(defgrad0).reshape(3, 3))

        # Maybe we need to avoid a negative determinant?

        # AP: something to do with avoiding zeros on
        # the diagonal, perhaps presuming the zero was introduced in the above step:
        if defgrad0[0] == 0.0:
            defgrad0[0] = det0 / (defgrad0[4] * defgrad0[8] - defgrad0[5] * defgrad0[7])

        if defgrad0[4] == 0.0:
            defgrad0[4] = det0 / (defgrad0[0] * defgrad0[8] - defgrad0[2] * defgrad0[6])

        if defgrad0[8] == 0.0:
            defgrad0[8] = det0 / (defgrad0[0] * defgrad0[4] - defgrad0[1] * defgrad0[3])

        # AP: compute Green strain
        strain = 0.5 * (np.dot(np.array(defgrad0).reshape(3, 3).T,
                               np.array(defgrad0).reshape(3, 3)) - np.eye(3))  # Green Strain

        # AP: compute Von Mises equivalent strain (i.e. a scalar to compare with
        # specified final strain.):
        eqstrain = 2.0 / 3.0 * np.sqrt(1.5 * (strain[0][0]**2 + strain[1][1]**2 + strain[2][2]**2) +
                                       3.0 * (strain[0][1]**2 + strain[1][2]**2 + strain[2][0]**2))

        # AP: factor of 1.05 to make sure we go just over the requested final strain?
        ratio = finalStrain * 1.05 / eqstrain

        # AP: the returned scale factor should be >= 1
        return max(ratio, 1.0)

    all_load_cases = []
    for total_time_i, num_incs_i, target_strain_i in zip(total_time, num_increments, target_strain):

        # print('total_time_i: {}'.format(total_time_i))
        # print('num_incs_i: {}'.format(num_incs_i))
        # print('target_strain_i: {}'.format(target_strain_i))

        defgrad = ['*'] * 9
        stress = [0] * 9
        values = (np.random.random_sample(9) - .5) * target_strain_i * 2

        main = np.array([0, 4, 8])  # AP: these are the diagonal element indices
        np.random.shuffle(main)

        # fill 2 out of 3 main entries
        for i in main[:2]:
            defgrad[i] = 1. + values[i]
            stress[i] = '*'

        # fill 3 off-diagonal pairs of defgrad (1 or 2 entries)
        for off in [[1, 3, 0], [2, 6, 0], [5, 7, 0]]:
            off = np.array(off)
            np.random.shuffle(off)
            for i in off[0:2]:
                if i != 0:
                    defgrad[i] = values[i]
                    stress[i] = '*'

        ratio = defgradScale(defgrad, target_strain_i)

        # AP: scale all elements by ratio (disregarding the identity on the
        # diagonal.)
        for i in [0, 4, 8]:
            if defgrad[i] != '*':
                defgrad[i] = (defgrad[i] - 1.0) * ratio + 1.0

        for i in [1, 2, 3, 5, 6, 7]:
            if defgrad[i] != '*':
                defgrad[i] = defgrad[i] * ratio

        dg_arr = []
        mask_arr = []
        for i in range(3):
            dg_arr_row = []
            mask_arr_row = []
            for j in range(3):
                idx = 3*i + j
                dg_arr_val = 0 if defgrad[idx] == '*' else defgrad[idx]
                dg_arr_row.append(dg_arr_val)
                mask_arr_val = 1 if defgrad[idx] == '*' else 0
                mask_arr_row.append(mask_arr_val)
            dg_arr.append(dg_arr_row)
            mask_arr.append(mask_arr_row)

        dg_arr = np.ma.masked_array(dg_arr, mask=mask_arr)

        stress_arr = []
        mask_arr = []
        for i in range(3):
            stress_arr_row = []
            mask_arr_row = []
            for j in range(3):
                idx = 3*i + j
                stress_arr_val = 0 if stress[idx] == '*' else stress[idx]
                stress_arr_row.append(stress_arr_val)
                mask_arr_val = 1 if stress[idx] == '*' else 0
                mask_arr_row.append(mask_arr_val)
            stress_arr.append(stress_arr_row)
            mask_arr.append(mask_arr_row)

        stress_arr = np.ma.masked_array(stress_arr, mask=mask_arr)

        load_case = {
            'total_time': total_time_i,
            'num_increments': num_incs_i,
            'def_grad_aim': dg_arr,
            'stress': stress_arr,
        }
        all_load_cases.append(load_case)

    return all_load_cases


def get_random_3D_def_grad(magnitude):

    # Find a random rotation matrix and a random stretch matrix and
    # multiply: F = RU

    R = get_random_rotation_matrix()

    # Five stretch components, since it's a symmetric matrix and the
    # trace must be zero:
    stretch_comps = np.random.random((5,)) * magnitude
    stretch = np.zeros((3, 3)) * np.nan

    # Diagonal comps:
    stretch[[0, 1], [0, 1]] = stretch_comps[:2]
    stretch[2, 2] = -(stretch[0, 0] + stretch[1, 1])

    # Off-diagonal comps:
    stretch[[1, 0], [0, 1]] = stretch_comps[2]
    stretch[[2, 0], [0, 2]] = stretch_comps[3]
    stretch[[1, 2], [2, 1]] = stretch_comps[4]

    # Add the identity:
    U = stretch + np.eye(3)

    # Scale by the determinant:
    U = U / np.linalg.det(U)

    F = R @ U

    return F


def get_random_3D_def_grad_v2(magnitude):

    # Find a random rotation matrix and a random stretch matrix and
    # multiply: F = RU

    R = get_random_rotation_matrix()
    print('R:\n{}'.format(R))

    assert np.allclose(R.T @ R, np.eye(3))
    assert np.isclose(np.linalg.det(R), 1)

    # Five stretch components, since it's a symmetric matrix and the
    # trace must be zero:
    stretch_comps = (np.random.random((5,)) - 0.5) * magnitude

    stretch = np.zeros((3, 3)) * np.nan

    # Diagonal comps (trace must be zero):
    stretch[[0, 1], [0, 1]] = stretch_comps[:2]
    stretch[2, 2] = -(stretch[0, 0] + stretch[1, 1])

    # Off-diagonal comps:
    stretch[[1, 0], [0, 1]] = stretch_comps[2]
    stretch[[2, 0], [0, 2]] = stretch_comps[3]
    stretch[[1, 2], [2, 1]] = stretch_comps[4]

    print('\nstretch:\n{}'.format(stretch))

    # Add the identity:
    U = stretch

    print('\nU:\n{}'.format(U))

    F = (R @ U) + np.eye(3)

    F_det = np.linalg.det(F)
    print('\nF det: {}'.format(F_det))

    F = F / F_det

    return F


def get_load_case_random_3d_v2(total_time, num_increments, target_strain, rotation=True,
                               rotation_max_angle=10, rotation_load_case=True):

    all_load_cases = []
    for total_time_i, num_incs_i, target_strain_i in zip(
            total_time, num_increments, target_strain):

        # Five stretch components, since it's a symmetric matrix and the
        # trace must be zero:
        stretch_comps = (np.random.random((5,)) - 0.5) * target_strain_i
        stretch = np.zeros((3, 3)) * np.nan

        # Diagonal comps:
        stretch[[0, 1], [0, 1]] = stretch_comps[:2]
        stretch[2, 2] = -(stretch[0, 0] + stretch[1, 1])

        # Off-diagonal comps:
        stretch[[1, 0], [0, 1]] = stretch_comps[2]
        stretch[[2, 0], [0, 2]] = stretch_comps[3]
        stretch[[1, 2], [2, 1]] = stretch_comps[4]

        # Add the identity:
        U = stretch + np.eye(3)

        defgrad = U
        rot = None
        if rotation:
            rot = get_random_rotation_matrix(method='axis_angle',
                                             max_angle_deg=rotation_max_angle)
            if not rotation_load_case:
                defgrad = rot @ U
                rot = None

        # Ensure defgrad has a unit determinant:
        defgrad = defgrad / (np.linalg.det(defgrad)**(1/3))

        dg_arr = np.ma.masked_array(defgrad, mask=np.zeros((3, 3), dtype=int))
        stress_arr = np.ma.masked_array(
            np.zeros((3, 3), dtype=int),
            mask=np.ones((3, 3), dtype=int)
        )

        # ===========
        # dg_arr.mask[[0, 0, 1], [0, 1, 0]] = 1
        # stress_arr.mask[[0, 0, 1], [0, 1, 0]] = 0
        # ===========

        load_case = {
            'total_time': total_time_i,
            'num_increments': num_incs_i,
            'def_grad_aim': dg_arr,
            'stress': stress_arr,
            'rotation': rot,
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


def generate_geom_VTK(path, volume_element):
    geom_path = path.parent.joinpath('geom.geom')
    write_damask_geom(geom_path, volume_element)


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
        'geom.vtk': generate_geom_VTK,
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

TASK_FUNC_MAP.update({
    ('generate_load_case', 'uniaxial'): get_load_case_uniaxial,
    ('generate_load_case', 'biaxial'): get_load_case_biaxial,
    ('generate_load_case', 'plane_strain'): get_load_case_plane_strain,
    ('generate_load_case', 'random_2d'): get_load_case_random_2d,
    ('generate_load_case', 'random_3d'): get_load_case_random_3d,
    ('generate_load_case', 'random_3d_v2'): get_load_case_random_3d_v2,
})
