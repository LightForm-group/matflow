"""Module containing unit tests on Task logic."""

import copy
import unittest

from matflow.models import TaskSchema
from matflow.models.construction import normalise_local_inputs, get_local_inputs
from matflow.errors import (
    IncompatibleSequence,
    TaskSchemaError,
    TaskParameterError,
    SequenceError,
)

# TODO: add test that warn is issued when an input is in base but also has a sequence.


class TaskSchemaTestCase(unittest.TestCase):
    """Tests on TaskSchema"""

    def test_raise_on_input_is_output(self):
        with self.assertRaises(TaskSchemaError):
            TaskSchema('schema_1', inputs=['parameter_1'], outputs=['parameter_1'])

    def test_raise_on_input_map_bad_inputs(self):
        """Check inputs defined in the schema input map are in the schema inputs list."""

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
        """Check outputs defined in the schema output map are in the schema outputs list."""

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
    """Tests of correct behaviour when defining tasks."""

    def test_raise_on_unknown_input(self):
        with self.assertRaises(TaskParameterError):
            schema = TaskSchema(
                'schema_1',
                inputs=['parameter_1'],
                outputs=['parameter_2'],
            )
            schema.check_surplus_inputs(['parameter_3'])

    def test_raise_on_missing_input(self):
        with self.assertRaises(TaskParameterError):
            schema = TaskSchema(
                'schema1',
                inputs=['parameter_1', 'parameter_2'],
                outputs=['parameter_3'],
            )
            schema.check_missing_inputs(['parameter_2'])


class NormaliseLocalTestCase(unittest.TestCase):
    """Testing `normalise_local_inputs`."""

    def test_raise_on_bad_nest_idx_float(self):
        """Check raises on non-integer (float) nest index for any sequence."""
        sequences = [{'name': 'p1', 'nest_idx': 1.0, 'vals': [101, 102]}]
        with self.assertRaises(SequenceError):
            normalise_local_inputs(sequences=sequences)

    def test_raise_on_bad_nest_idx_string(self):
        """Check raises on non-integer (str) nest index for any sequence."""
        sequences = [{'name': 'p1', 'nest_idx': '0', 'vals': [101, 102]}]
        with self.assertRaises(SequenceError):
            normalise_local_inputs(sequences=sequences)

    def test_raise_on_bad_nest_idx_list(self):
        """Check raises on non-integer (list) nest index for any sequence."""
        sequences = [{'name': 'p1', 'nest_idx': [1, 0], 'vals': [101, 102]}]
        with self.assertRaises(SequenceError):
            normalise_local_inputs(sequences=sequences)


class GetLocalInputsExceptionTestCase(unittest.TestCase):
    """Testing exceptions and warnings from `get_local_inputs`."""

    def test_raise_on_missing_nest_idx(self):
        """Check raises when more than one sequence, but nest_idx is missing from any
        sequence."""
        sequences = [
            {'name': 'p2', 'vals': [201, 202], 'nest_idx': 0},
            {'name': 'p3', 'vals': [301, 302]},
        ]
        with self.assertRaises(SequenceError):
            get_local_inputs([], sequences=sequences)

    def test_raise_on_bad_sequence_vals_type_str(self):
        """Test raises when sequence vals is a string."""
        sequences = [{'name': 'p1', 'vals': '120'}]
        with self.assertRaises(SequenceError):
            get_local_inputs([], sequences=sequences)

    def test_raise_on_bad_sequence_vals_type_number(self):
        """Test raises when sequence vals is a number."""
        sequences = [{'name': 'p1', 'vals': 120}]
        with self.assertRaises(SequenceError):
            get_local_inputs([], sequences=sequences)

    def test_raise_on_bad_sequences_type(self):
        """Test raises when sequences is not a list."""
        sequences = {'name': 'p1', 'vals': [1, 2]}
        with self.assertRaises(SequenceError):
            get_local_inputs([], sequences=sequences)

    def test_warn_on_unrequired_nest_idx(self):
        """Test warning on unrequired nest idx."""
        sequences = [{'name': 'p1', 'vals': [101, 102], 'nest_idx': 0}]
        with self.assertWarns(Warning):
            get_local_inputs([], sequences=sequences)

    def test_raise_on_bad_sequence_keys(self):
        """Test raises when a sequence has unknown keys."""
        sequences = [{'name': 'p1', 'vals': [101, 102], 'bad_key': 4}]
        with self.assertRaises(SequenceError):
            get_local_inputs([], sequences=sequences)

    def test_raise_on_missing_sequence_keys(self):
        """Test raises when a sequence has missing keys."""
        sequences = [{'vals': [101, 102]}]
        with self.assertRaises(SequenceError):
            get_local_inputs([], sequences=sequences)

    def test_raise_on_incompatible_nesting(self):
        """Test error raised on logically inconsistent Task sequence."""
        sequences = [
            {'name': 'p1', 'nest_idx': 0, 'vals': [101, 102]},
            {'name': 'p2', 'nest_idx': 0, 'vals': [201]},
        ]
        with self.assertRaises(IncompatibleSequence):
            get_local_inputs([], sequences=sequences)


class GetLocalInputsInputsTestCase(unittest.TestCase):
    """Tests on the `inputs` dict generated by `get_local_inputs`."""

    def test_base_only(self):
        """Check expected output for no sequences."""
        base = {'p1': 101}
        local_ins = get_local_inputs([], base=base)['inputs']
        local_ins_exp = {'p1': {'vals': [101], 'vals_idx': [0]}}
        self.assertTrue(local_ins == local_ins_exp)

    def test_base_and_sequence(self):
        """Check expected output for base and one sequence."""
        base = {'p1': 101}
        sequences = [{'name': 'p2', 'vals': [201, 202]}]
        local_ins = get_local_inputs([], base=base, sequences=sequences)['inputs']
        local_ins_exp = {
            'p1': {'vals': [101], 'vals_idx': [0, 0]},
            'p2': {'vals': [201, 202], 'vals_idx': [0, 1]},
        }
        self.assertTrue(local_ins == local_ins_exp)

    def test_base_and_multi_nested_sequences(self):
        """Check expected output for base and two nested sequences."""
        base = {'p1': 101}
        sequences = [
            {'name': 'p2', 'vals': [201, 202], 'nest_idx': 0},
            {'name': 'p3', 'vals': [301, 302, 303], 'nest_idx': 1},
        ]
        local_ins = get_local_inputs([], base=base, sequences=sequences)['inputs']
        local_ins_exp = {
            'p1': {'vals': [101], 'vals_idx': [0, 0, 0, 0, 0, 0]},
            'p2': {'vals': [201, 202], 'vals_idx': [0, 0, 0, 1, 1, 1]},
            'p3': {'vals': [301, 302, 303], 'vals_idx': [0, 1, 2, 0, 1, 2]},
        }
        self.assertTrue(local_ins == local_ins_exp)

    def test_base_and_multi_merged_sequences(self):
        """Check expected output for base and two merged sequences."""
        base = {'p1': 101}
        sequences = [
            {'name': 'p2', 'vals': [201, 202], 'nest_idx': 0},
            {'name': 'p3', 'vals': [301, 302], 'nest_idx': 0},
        ]
        local_ins = get_local_inputs([], base=base, sequences=sequences)['inputs']
        local_ins_exp = {
            'p1': {'vals': [101], 'vals_idx': [0, 0]},
            'p2': {'vals': [201, 202], 'vals_idx': [0, 1]},
            'p3': {'vals': [301, 302], 'vals_idx': [0, 1]},
        }
        self.assertTrue(local_ins == local_ins_exp)

    def test_base_and_merged_and_nested_sequences(self):
        """Check expected output for base and two merged sequences."""
        base = {'p1': 101}
        sequences = [
            {'name': 'p2', 'vals': [201, 202], 'nest_idx': 0},
            {'name': 'p3', 'vals': [301, 302], 'nest_idx': 0},
            {'name': 'p4', 'vals': [401, 402, 403], 'nest_idx': 1},
        ]
        local_ins = get_local_inputs([], base=base, sequences=sequences)['inputs']
        local_ins_exp = {
            'p1': {'vals': [101], 'vals_idx': [0, 0, 0, 0, 0, 0]},
            'p2': {'vals': [201, 202], 'vals_idx': [0, 0, 0, 1, 1, 1]},
            'p3': {'vals': [301, 302], 'vals_idx': [0, 0, 0, 1, 1, 1]},
            'p4': {'vals': [401, 402, 403], 'vals_idx': [0, 1, 2, 0, 1, 2]},
        }
        self.assertTrue(local_ins == local_ins_exp)

    def test_equivalent_relative_nesting_idx(self):
        """Check the actual value of `nest_idx` is inconsequential."""
        sequences_1 = [
            {'name': 'p1', 'nest_idx': 0, 'vals': [101, 102, 103]},
            {'name': 'p2', 'nest_idx': 1, 'vals': [201, 202]},
        ]
        sequences_2 = copy.deepcopy(sequences_1)
        sequences_2[0]['nest_idx'] = 105
        sequences_2[1]['nest_idx'] = 2721

        local_ins_1 = get_local_inputs([], sequences=sequences_1)['inputs']
        local_ins_2 = get_local_inputs([], sequences=sequences_2)['inputs']

        self.assertTrue(local_ins_1 == local_ins_2)

    def test_correct_number_of_local_inputs_all_nesting(self):
        """Check the correct number of elements for a given input."""
        sequences = [
            {'name': 'p1', 'nest_idx': 0, 'vals': [101, 102, 103]},
            {'name': 'p2', 'nest_idx': 1, 'vals': [201, 202]},
        ]
        local_ins = get_local_inputs([], sequences=sequences)['inputs']
        self.assertTrue(len(local_ins['p1']['vals_idx']) == 6)

    def test_all_inputs_local_inputs_size(self):
        """Check all inputs have the same number of elements."""
        sequences = [
            {'name': 'p1', 'nest_idx': 0, 'vals': [101, 102, 103]},
            {'name': 'p2', 'nest_idx': 1, 'vals': [201, 202]},
        ]
        local_ins = get_local_inputs([], sequences=sequences)['inputs']
        self.assertTrue(
            len(local_ins['p1']['vals_idx']) == len(local_ins['p2']['vals_idx'])
        )

    def test_correct_number_of_local_inputs_all_merge(self):
        """Check the correct number of local inputs for merging three sequences."""
        sequences = [
            {'name': 'p1', 'nest_idx': 3, 'vals': [101, 102]},
            {'name': 'p2', 'nest_idx': 3, 'vals': [201, 202]},
            {'name': 'p3', 'nest_idx': 3, 'vals': [301, 302]},
        ]
        local_ins = get_local_inputs([], sequences=sequences)['inputs']
        self.assertTrue(
            len(local_ins['p1']['vals_idx']) ==
            len(local_ins['p2']['vals_idx']) ==
            len(local_ins['p3']['vals_idx']) == 2
        )

    def test_correct_number_of_local_inputs_one_merge(self):
        """Check the correct number of local inputs for merging/nesting three sequences."""
        sequences = [
            {'name': 'p1', 'nest_idx': 3, 'vals': [101, 102]},
            {'name': 'p2', 'nest_idx': 4, 'vals': [201, 202]},
            {'name': 'p3', 'nest_idx': 4, 'vals': [301, 302]},
        ]
        local_ins = get_local_inputs([], sequences=sequences)['inputs']
        self.assertTrue(
            len(local_ins['p1']['vals_idx']) ==
            len(local_ins['p2']['vals_idx']) ==
            len(local_ins['p3']['vals_idx']) == 4
        )

    def test_base_is_merged_into_sequence(self):
        """Check the base dict is merged into a sequence."""
        base = {'p1': 101}
        sequences = [{'name': 'p2', 'nest_idx': 0, 'vals': [201, 202]}]
        local_ins = get_local_inputs([], base=base, sequences=sequences)['inputs']
        self.assertTrue(
            local_ins['p1']['vals_idx'] == [0, 0] and
            local_ins['p2']['vals_idx'] == [0, 1]
        )

    def test_unit_length_sequence(self):
        """Check specifying sequences of length one has the same effect as specifying the 
        parameter in the base dict."""
        base = {'p1': 101}
        sequences = [{'name': 'p1', 'nest_idx': 0, 'vals': [101]}]
        local_ins_1 = get_local_inputs([], sequences=sequences)['inputs']
        local_ins_2 = get_local_inputs([], base=base)['inputs']
        self.assertTrue(local_ins_1 == local_ins_2)


class GetLocalInputsFullTestCase(unittest.TestCase):
    """Explicit checks on the full outputs of `get_local_inputs`."""

    def full_test_1(self):
        pass
