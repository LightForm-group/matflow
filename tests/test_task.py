"""Module containing unit tests on Task logic."""

import copy
import unittest

from matflow.models.task import resolve_local_inputs, TaskSchema, get_local_inputs
from matflow.errors import IncompatibleSequence, TaskSchemaError, TaskParameterError

# TODO: add test that warn is issued when an input is in base but also has a sequence.


class TaskSchemaTestCase(unittest.TestCase):
    'Tests on TaskSchema'

    def test_raise_on_input_is_output(self):
        with self.assertRaises(TaskSchemaError):
            TaskSchema('schema_1', inputs=['parameter_1'], outputs=['parameter_1'])

    def test_raise_on_input_map_bad_inputs(self):
        'Check inputs defined in the schema input map are in the schema inputs list.'

        with self.assertRaises(TaskSchemaError):
            TaskSchema(
                'schema_1',
                inputs=['parameter_7', 'parameter_9'],
                outputs=['parameter_8'],
                input_map=[
                    {
                        'inputs': [
                            # "parameter_10" is not in the inputs list.
                            'parameter_10',
                        ],
                        'file': 'input_file_1',
                    }
                ]
            )

    def test_raise_on_output_map_bad_outputs(self):
        'Check outputs defined in the schema output map are in the schema outputs list.'

        with self.assertRaises(TaskSchemaError):
            TaskSchema(
                'schema_1',
                inputs=['parameter_7', 'parameter_9'],
                outputs=['parameter_8'],
                output_map=[
                    {
                        'files': [
                            'output_file_1',
                        ],
                        # "parameter_10" is not in the outputs list.
                        'output': 'parameter_10',
                    }
                ]
            )


class TaskParameterTestCase(unittest.TestCase):
    'Tests of correct behaviour when defining tasks.'

    def test_raise_on_unknown_input(self):
        with self.assertRaises(TaskParameterError):
            schema = TaskSchema(
                'schema_1',
                inputs=['parameter_1'],
                outputs=['parameter_2'],
            )
            schema.check_surplus_inputs(['parameter_3'])

    # def test_raise_on_missing_input(self):
    #     with self.assertRaises(TaskParameterError):
    #         schema = TaskSchema(
    #             'schema1',
    #             inputs=['parameter_1', 'parameter_2'],
    #             outputs=['parameter_3'],
    #         )
    #         schema.check_surplus_inputs(['parameter_2'])


class ResolveLocalInputsTestCase(unittest.TestCase):
    'Tests on generating sequences with `resolve_local_inputs`.'

    # TODO: also do some explicit checks that input values are propagated properly.

    def test_base_only(self):
        'Check expected output for no sequences.'
        base = {
            'parameter_1': 101,
        }
        local_ins = resolve_local_inputs(base=base)
        local_ins_exp = [
            {
                'parameter_1': 101,
            }
        ]
        self.assertTrue(local_ins == local_ins_exp)

    def test_base_and_sequence(self):
        'Check expected output for base and one sequence.'

        # TODO: currently fails since nest_idx is (wrongly) always required.

        base = {
            'parameter_1': 101,
        }
        sequences = [
            {
                'name': 'parameter_2',
                'vals': [201, 202],
            }
        ]
        local_ins = resolve_local_inputs(base=base, sequences=sequences)
        local_ins_exp = [
            {
                'parameter_1': 101,
                'parameter_2': 201,
            },
            {
                'parameter_1': 101,
                'parameter_2': 202,
            },
        ]
        self.assertTrue(local_ins == local_ins_exp)

    def test_base_and_multi_nested_sequences(self):
        'Check expected output for base and two nested sequences.'
        base = {
            'parameter_1': 101,
        }
        sequences = [
            {
                'name': 'parameter_2',
                'vals': [201, 202],
                'nest_idx': 0,
            },
            {
                'name': 'parameter_3',
                'vals': [301, 302, 303],
                'nest_idx': 1,
            },
        ]
        local_ins = resolve_local_inputs(base=base, sequences=sequences)
        local_ins_exp = [
            {
                'parameter_1': 101,
                'parameter_2': 201,
                'parameter_3': 301,
            },
            {
                'parameter_1': 101,
                'parameter_2': 201,
                'parameter_3': 302,
            },
            {
                'parameter_1': 101,
                'parameter_2': 201,
                'parameter_3': 303,
            },
            {
                'parameter_1': 101,
                'parameter_2': 202,
                'parameter_3': 301,
            },
            {
                'parameter_1': 101,
                'parameter_2': 202,
                'parameter_3': 302,
            },
            {
                'parameter_1': 101,
                'parameter_2': 202,
                'parameter_3': 303,
            },
        ]
        self.assertTrue(local_ins == local_ins_exp)

    def test_base_and_multi_merged_sequences(self):
        'Check expected output for base and two merged sequences.'
        base = {
            'parameter_1': 101,
        }
        sequences = [
            {
                'name': 'parameter_2',
                'vals': [201, 202],
                'nest_idx': 0,
            },
            {
                'name': 'parameter_3',
                'vals': [301, 302],
                'nest_idx': 0,
            },
        ]
        local_ins = resolve_local_inputs(base=base, sequences=sequences)
        local_ins_exp = [
            {
                'parameter_1': 101,
                'parameter_2': 201,
                'parameter_3': 301,
            },
            {
                'parameter_1': 101,
                'parameter_2': 202,
                'parameter_3': 302,
            },
        ]
        self.assertTrue(local_ins == local_ins_exp)

    def test_base_and_merged_and_nested_sequences(self):
        'Check expected output for base and two merged sequences.'
        base = {
            'parameter_1': 101,
        }
        sequences = [
            {
                'name': 'parameter_2',
                'vals': [201, 202],
                'nest_idx': 0,
            },
            {
                'name': 'parameter_3',
                'vals': [301, 302],
                'nest_idx': 0,
            },
            {
                'name': 'parameter_4',
                'vals': [401, 402, 403],
                'nest_idx': 1,
            },
        ]
        local_ins = resolve_local_inputs(base=base, sequences=sequences)
        local_ins_exp = [
            {
                'parameter_1': 101,
                'parameter_2': 201,
                'parameter_3': 301,
                'parameter_4': 401,
            },
            {
                'parameter_1': 101,
                'parameter_2': 201,
                'parameter_3': 301,
                'parameter_4': 402,
            },
            {
                'parameter_1': 101,
                'parameter_2': 201,
                'parameter_3': 301,
                'parameter_4': 403,
            },
            {
                'parameter_1': 101,
                'parameter_2': 202,
                'parameter_3': 302,
                'parameter_4': 401,
            },
            {
                'parameter_1': 101,
                'parameter_2': 202,
                'parameter_3': 302,
                'parameter_4': 402,
            },
            {
                'parameter_1': 101,
                'parameter_2': 202,
                'parameter_3': 302,
                'parameter_4': 403,
            },
        ]
        self.assertTrue(local_ins == local_ins_exp)

    def test_raise_on_missing_nest_idx(self):
        """Check raises when more than one sequence, but nest_idx is missing from any
        sequence."""

    def test_raise_on_bad_sequence_vals_type(self):
        'i.e. not a list.'
        pass

    def test_raise_on_bad_sequences_type(self):
        'i.e. not a list.'
        pass

    def test_warn_on_unrequired_nest_idx(self):
        pass

    def test_raise_on_bad_nest_idx_float(self):
        'Check raises on non-integer (float) nest index for any sequence.'
        sequences = [
            {
                'name': 'parameter_1',
                'nest_idx': 1.0,
                'vals': [101, 102],
            },
        ]
        with self.assertRaises(ValueError):
            _ = resolve_local_inputs(sequences=sequences)

    def test_raise_on_bad_nest_idx_string(self):
        'Check raises on non-integer (str) nest index for any sequence.'
        sequences = [
            {
                'name': 'parameter_1',
                'nest_idx': '0',
                'vals': [101, 102],
            },
        ]
        with self.assertRaises(ValueError):
            _ = resolve_local_inputs(sequences=sequences)

    def test_raise_on_bad_nest_idx_list(self):
        'Check raises on non-integer (str) nest index for any sequence.'
        sequences = [
            {
                'name': 'parameter_1',
                'nest_idx': [1, 0],
                'vals': [101, 102],
            },
        ]
        with self.assertRaises(ValueError):
            _ = resolve_local_inputs(sequences=sequences)

    def test_equivalent_relative_nesting_idx(self):
        'Check the actual value of `nest_idx` is inconsequential.'
        sequences_1 = [
            {
                'name': 'parameter_1',
                'nest_idx': 0,
                'vals': [101, 102, 103],
            },
            {
                'name': 'parameter_2',
                'nest_idx': 1,
                'vals': [201, 202],
            },
        ]
        sequences_2 = copy.deepcopy(sequences_1)
        sequences_2[0]['nest_idx'] = 105
        sequences_2[1]['nest_idx'] = 2721

        local_ins_1 = resolve_local_inputs(sequences=sequences_1)
        local_ins_2 = resolve_local_inputs(sequences=sequences_2)

        self.assertTrue(local_ins_1 == local_ins_2)

    def test_correct_number_of_local_inputs_all_nesting(self):
        sequences = [
            {
                'name': 'parameter_1',
                'nest_idx': 0,
                'vals': [101, 102, 103],
            },
            {
                'name': 'parameter_2',
                'nest_idx': 1,
                'vals': [201, 202],
            },
        ]
        local_ins = resolve_local_inputs(sequences=sequences)
        self.assertTrue(len(local_ins) == 6)

    def test_correct_number_of_local_inputs_all_merge(self):
        sequences = [
            {
                'name': 'parameter_1',
                'nest_idx': 3,
                'vals': [101, 102],
            },
            {
                'name': 'parameter_2',
                'nest_idx': 3,
                'vals': [201, 202],
            },
            {
                'name': 'parameter_3',
                'nest_idx': 3,
                'vals': [301, 302],
            },
        ]
        local_ins = resolve_local_inputs(sequences=sequences)
        self.assertTrue(len(local_ins) == 2)

    def test_correct_number_of_local_inputs_one_merge(self):
        # TODO: need to test merge order?

        sequences = [
            {
                'name': 'parameter_1',
                'nest_idx': 3,
                'vals': [101, 102],
            },
            {
                'name': 'parameter_2',
                'nest_idx': 4,
                'vals': [201, 202],
            },
            {
                'name': 'parameter_3',
                'nest_idx': 4,
                'vals': [301, 302],
            },
        ]
        local_ins = resolve_local_inputs(sequences=sequences)
        self.assertTrue(len(local_ins) == 2)

    def test_base_is_merged_into_sequence(self):
        'Check the base dict is merged into a sequence.'
        base = {
            'parameter_1': 101
        }
        sequences = [
            {
                'name': 'parameter_2',
                'nest_idx': 0,
                'vals': [201, 202],
            },
        ]
        local_ins = resolve_local_inputs(base=base, sequences=sequences)
        self.assertTrue(
            local_ins[0]['parameter_1'] == 101 and
            local_ins[1]['parameter_1'] == 101
        )

    def test_raise_on_incompatible_nesting(self):
        'Test error raised on logically inconsistent Task sequence.'
        sequences = [
            {
                'name': 'parameter_1',
                'nest_idx': 0,
                'vals': [101, 102],
            },
            {
                'name': 'parameter_2',
                'nest_idx': 0,
                'vals': [201],
            },
        ]
        with self.assertRaises(IncompatibleSequence):
            _ = resolve_local_inputs(sequences=sequences)

    def test_unit_length_sequence(self):
        """Check specifying sequences of length one has the same effect as specifying the 
        parameter in the base dict."""

        # Currently fails due to `base` params being assigned `nest_idx=-1`

        sequences = [
            {
                'name': 'parameter_1',
                'nest_idx': 0,
                'vals': [101],
            },
        ]
        base = {
            'parameter_1': 101
        }
        local_ins_1 = get_local_inputs(sequences=sequences)
        local_ins_2 = get_local_inputs(base=base)

        print(f'local_ins_1: {local_ins_1}')
        print(f'local_ins_2: {local_ins_2}')

        self.assertTrue(local_ins_1 == local_ins_2)

    def test_sequence_repeats(self):

        sequences = [
            {
                'name': 'parameter_1',
                'nest_idx': 0,
                'vals': [101, 102],
            },
            {
                'repeats': 2,
            }
        ]
        local_ins = get_local_inputs(sequences=sequences)['inputs']

        # Expand `vals_idx`:
        local_ins = {k: np.array(v['vals'])[v['vals_idx']] for k, v in local_ins.items()}

        print(f'local_ins: {local_ins}')
