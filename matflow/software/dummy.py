"""`matflow.software.dummy.py

Dummy software for testing/docs.

"""

from pathlib import Path
from textwrap import dedent
from random import randint

from matflow import TASK_INPUT_MAP, TASK_OUTPUT_MAP, TASK_FUNC_MAP, COMMAND_LINE_ARG_MAP


def dummy_input_map_1(path, parameter_1):
    with Path(path).open('w') as handle:
        handle.write('{}\n'.format(randint(0, 1e6)))
        handle.write('parameter_1: {}\n'.format(parameter_1))


def dummy_input_map_1_inv(path, parameter_2):
    with Path(path).open('w') as handle:
        handle.write('{}\n'.format(randint(0, 1e6)))
        handle.write('parameter_2: {}\n'.format(parameter_2))


def dummy_input_map_2(path, parameter_2, parameter_3):
    with Path(path).open('w') as handle:
        handle.write('{}\n'.format(randint(0, 1e6)))
        handle.write('parameter_2: {}\n'.format(parameter_2))
        handle.write('parameter_3: {}\n'.format(parameter_3))


def dummy_input_map_2b(path, parameter_2, parameter_3):
    with Path(path).open('w') as handle:
        handle.write('{}\n'.format(randint(0, 1e6)))
        handle.write('parameter_2: {}\n'.format(parameter_2))
        handle.write('parameter_3: {}\n'.format(parameter_3))


def dummy_input_map_3(path, parameter_5):
    with Path(path).open('w') as handle:
        handle.write('{}\n'.format(randint(0, 1e6)))
        handle.write('parameter_5: {}\n'.format(parameter_5))


def dummy_input_map_4(path, parameter_2, parameter_6, parameter_7, parameter_9):
    with Path(path).open('w') as handle:
        handle.write('{}\n'.format(randint(0, 1e6)))
        handle.write('parameter_2: {}\n'.format(parameter_2))
        handle.write('parameter_6: {}\n'.format(parameter_6))
        handle.write('parameter_7: {}\n'.format(parameter_7))
        handle.write('parameter_9: {}\n'.format(parameter_9))


def dummy_input_map_5(path, parameter_8_group, parameter_10):
    with Path(path).open('w') as handle:
        handle.write('{}\n'.format(randint(0, 1e6)))
        handle.write('parameter_8_group: {}\n'.format(parameter_8_group))
        handle.write('parameter_10: {}\n'.format(parameter_10))


def dummy_input_map_5b(path, parameter_8A, parameter_8B):
    with Path(path).open('w') as handle:
        handle.write('{}\n'.format(randint(0, 1e6)))
        handle.write('parameter_8A: {}\n'.format(parameter_8A))
        handle.write('parameter_8B: {}\n'.format(parameter_8B))


def dummy_input_map_5c(path, parameter_8A_group, parameter_8B_group, parameter_10):
    with Path(path).open('w') as handle:
        handle.write('{}\n'.format(randint(0, 1e6)))
        handle.write('parameter_8A_group: {}\n'.format(parameter_8A_group))
        handle.write('parameter_8B_group: {}\n'.format(parameter_8B_group))
        handle.write('parameter_10: {}\n'.format(parameter_10))


def dummy_input_map_6(path, parameter_11, parameter_12):
    with Path(path).open('w') as handle:
        handle.write('{}\n'.format(randint(0, 1e6)))
        handle.write('parameter_11: {}\n'.format(parameter_11))
        handle.write('parameter_12: {}\n'.format(parameter_12))


def dummy_input_map_6b(path, parameter_4_multiaxial, parameter_4_uniaxial, parameter_5):
    with Path(path).open('w') as handle:
        handle.write('{}\n'.format(randint(0, 1e6)))
        handle.write('parameter_4_uniaxial: {}\n'.format(parameter_4_uniaxial))
        handle.write('parameter_4_multiaxial: {}\n'.format(parameter_4_multiaxial))
        handle.write('parameter_5: {}\n'.format(parameter_5))


def dummy_output_map_1(path):

    with Path(path).open('r') as handle:
        parameter_2 = int(handle.readline().strip())

    return parameter_2


def dummy_output_map_2(path):

    with Path(path).open('r') as handle:
        parameter_4 = int(handle.readline().strip())

    return parameter_4


def dummy_output_map_3(path):

    with Path(path).open('r') as handle:
        parameter_6 = int(handle.readline().strip())

    return parameter_6


def dummy_output_map_4(path):

    with Path(path).open('r') as handle:
        parameter_8 = int(handle.readline().strip())

    return parameter_8


def dummy_output_map_5(path):

    with Path(path).open('r') as handle:
        parameter_11 = int(handle.readline().strip())

    return parameter_11


def dummy_output_map_6b(path):

    with Path(path).open('r') as handle:
        parameter_8 = int(handle.readline().strip())

    return parameter_8


def fmt_parameter_1(parameter_1):
    return '{}'.format(parameter_1)


COMMAND_LINE_ARG_MAP.update({
    ('dummy_task_1', 'method_1', 'software_1'): {
        'parameter_1': fmt_parameter_1,
    }
})

TASK_INPUT_MAP.update({
    ('dummy_task_1', 'method_1', 'software_1'): {
        't1_m1_infile_1': dummy_input_map_1,
    },
    ('dummy_task_1_inv', 'method_1', 'software_1'): {
        't1_m1_infile_1': dummy_input_map_1_inv,
    },
    ('dummy_task_2', 'method_1', 'software_1'): {
        't2_m1_infile_1': dummy_input_map_2,
    },
    ('dummy_task_2b', 'method_1', 'software_1'): {
        't2b_m1_infile_1': dummy_input_map_2b,
    },
    ('dummy_task_3', 'method_1', 'software_1'): {
        't3_m1_infile_1': dummy_input_map_3,
    },
    ('dummy_task_4', 'method_1', 'software_1'): {
        't4_m1_infile_1': dummy_input_map_4,
    },
    ('dummy_task_5', 'method_1', 'software_1'): {
        't5_m1_infile_1': dummy_input_map_5,
    },
    ('dummy_task_5b', 'method_1', 'software_1'): {
        't5b_m1_infile_1': dummy_input_map_5b,
    },
    ('dummy_task_5c', 'method_1', 'software_1'): {
        't5c_m1_infile_1': dummy_input_map_5c,
    },
    ('dummy_task_6', 'method_1', 'software_1'): {
        't6_m1_infile_1': dummy_input_map_6,
    },
    ('dummy_task_6b', 'method_1', 'software_1'): {
        't6b_m1_infile_1': dummy_input_map_6b,
    },
})

TASK_OUTPUT_MAP.update({
    ('dummy_task_1', 'method_1', 'software_1'): {
        'parameter_2': dummy_output_map_1,
    },
    ('dummy_task_2', 'method_1', 'software_1'): {
        'parameter_4': dummy_output_map_2,
    },
    ('dummy_task_3', 'method_1', 'software_1'): {
        'parameter_6': dummy_output_map_3,
    },
    ('dummy_task_4', 'method_1', 'software_1'): {
        'parameter_8': dummy_output_map_4,
    },
    ('dummy_task_5', 'method_1', 'software_1'): {
        'parameter_11': dummy_output_map_5,
    },
    ('dummy_task_6b', 'method_1', 'software_1'): {
        'parameter_8': dummy_output_map_6b,
    },
})

# TASK_FUNC_MAP.update({
#     ('generate_load_case', 'uniaxial'): get_load_case_uniaxial,
# })
