"""`matflow.software.dummy.py

Dummy software for testing/docs.

"""

from pathlib import Path
from textwrap import dedent
from random import randint

from matflow import TASK_INPUT_MAP, TASK_OUTPUT_MAP, TASK_FUNC_MAP


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


def dummy_output_map_1(path):

    with Path(path).open('r') as handle:
        parameter_2 = int(handle.readline().strip())

    return parameter_2


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
})

TASK_OUTPUT_MAP.update({
    ('dummy_task_1', 'method_1', 'software_1'): {
        'parameter_2': dummy_output_map_1,
    },
})

# TASK_FUNC_MAP.update({
#     ('generate_load_case', 'uniaxial'): get_load_case_uniaxial,
# })
