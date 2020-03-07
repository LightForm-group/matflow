"""Module containing unit tests on Workflow initialisation."""

import unittest
from shutil import rmtree

from matflow import TEST_WORKFLOWS_DIR, TEST_WORKING_DIR
from matflow.api import make_workflow
from matflow.errors import (IncompatibleWorkflow, IncompatibleNesting,
                            MissingMergePriority)
from matflow.models.workflow import check_task_compatibility

"""
tests for inputs/outputs_idx:
- for a variety of scenarios, check all parameters from the same task have the same number of elements_idx.
- for a few scenarios, check expected elements_idx and task_idx.
- check all keys of output (i.e. `task_idx`) are exactly the set of task_idx values in downstream + upstream tasks.
- check works when no upstream tasks.

tests for resolve_task_num_elements:
- check works when no upstream tasks

"""


class TaskCompatibilityTestCase(unittest.TestCase):
    'Tests ensuring correct behaviour for incompatible tasks.'

    def tearDown(self):
        'Remove the generated directories.'
        for i in TEST_WORKING_DIR.glob('*'):
            if i.is_dir():
                rmtree(str(i))

    def test_output_non_exclusivity(self):
        """Ensure raises on a workflow that has multiple tasks that include the same
        output."""
        task_compat_props = [
            {
                'name': '',
                'inputs': ['parameter_1'],
                'length': 1,
                'nest_idx': 0,
                'outputs': ['parameter_2'],
            },
            {
                'name': '',
                'inputs': ['parameter_1'],
                'length': 1,
                'nest_idx': 0,
                'outputs': ['parameter_2'],
            },
        ]
        with self.assertRaises(IncompatibleWorkflow):
            check_task_compatibility(task_compat_props)

    def test_circular_reference(self):
        """Ensure raises on a workflow whose Tasks are circularly referential."""
        task_compat_props = [
            {
                'name': '',
                'inputs': ['parameter_1'],
                'length': 1,
                'nest_idx': 0,
                'outputs': ['parameter_2'],
            },
            {
                'name': '',
                'inputs': ['parameter_2'],
                'length': 1,
                'nest_idx': 0,
                'outputs': ['parameter_1'],
            },
        ]
        with self.assertRaises(IncompatibleWorkflow):
            check_task_compatibility(task_compat_props)


class TaskNestingTestCase(unittest.TestCase):
    'Tests ensuring expected task nesting behaviour.'

    def test_correct_num_elements_all_nesting(self):
        """Ensure correct number of elements on final task given all inputting tasks
        are nesting."""
        compat_props = [
            {
                'name': '',
                'inputs': ['parameter_1'],
                'outputs': ['parameter_2'],
                'length': 2,
                'nest': True,
            },
            {
                'name': '',
                'inputs': ['parameter_3'],
                'outputs': ['parameter_4'],
                'length': 3,
                'nest': True,
            },
            {
                'name': '',
                'inputs': ['parameter_2', 'parameter_4'],
                'outputs': ['parameter_5'],
                'length': 6,
            },
        ]
        num_elem_expected = [2, 3, 36]

        _, compat_props_new, _ = check_task_compatibility(compat_props)
        num_elem = [i['num_elements'] for i in compat_props_new]
        self.assertTrue(num_elem == num_elem_expected)

    def test_correct_num_elements_mixed_nesting(self):
        """Ensure correct number of elements on final task given mixed nesting of
        inputting tasks."""
        compat_props = [
            {
                'name': '',
                'inputs': ['parameter_1'],
                'outputs': ['parameter_2'],
                'length': 2,
                'nest': True,
                'merge_priority': 0,
            },
            {
                'name': '',
                'inputs': ['parameter_3'],
                'outputs': ['parameter_4'],
                'length': 6,
                'nest': False,
                'merge_priority': 1,
            },
            {
                'name': '',
                'inputs': ['parameter_2', 'parameter_4'],
                'outputs': ['parameter_5'],
                'length': 3,
            },
        ]
        num_elem_expected = [2, 6, 6]
        _, compat_props_new, _ = check_task_compatibility(compat_props)
        num_elem = [i['num_elements'] for i in compat_props_new]
        self.assertTrue(num_elem == num_elem_expected)

    def test_unit_num_elements(self):
        'Ensure for merging tasks with unit length, number of elements is one.'
        compat_props = [
            {
                'name': '',
                'inputs': ['parameter_1'],
                'outputs': ['parameter_2'],
                'length': 1,
                'nest': True,
            },
            {
                'name': '',
                'inputs': ['parameter_2'],
                'outputs': ['parameter_5'],
                'length': 1,
            },
        ]
        _, compat_props_1, _ = check_task_compatibility(compat_props)
        num_elem = [i['num_elements'] for i in compat_props_1]
        num_elem_expected = [1, 1]

        self.assertTrue(num_elem == num_elem_expected)

    def test_nesting_independence_for_unit_length_tasks(self):
        """Ensure for merging tasks with unit length, nesting value does not matter."""
        compat_props_1 = [
            {
                'name': '',
                'inputs': ['parameter_1'],
                'outputs': ['parameter_2'],
                'length': 1,
                'nest': True,
            },
            {
                'name': '',
                'inputs': ['parameter_2'],
                'outputs': ['parameter_5'],
                'length': 1,
            },
        ]
        compat_props_2 = [
            {
                'name': '',
                'inputs': ['parameter_1'],
                'outputs': ['parameter_2'],
                'length': 1,
                'nest': False,
            },
            {
                'name': '',
                'inputs': ['parameter_2'],
                'outputs': ['parameter_5'],
                'length': 1,
            },
        ]

        _, compat_props_1_new, _ = check_task_compatibility(compat_props_1)
        num_elem_1 = [i['num_elements'] for i in compat_props_1_new]

        _, compat_props_2_new, _ = check_task_compatibility(compat_props_2)
        num_elem_2 = [i['num_elements'] for i in compat_props_2_new]

        self.assertTrue(num_elem_1 == num_elem_2)

    def test_raise_on_missing_merge_priority(self):
        """Ensure raises if `merge_priority` is not specified on all tasks when there is a
        mix of `nest: True` and `nest: False` tasks."""
        compat_props = [
            {
                'name': '',
                'inputs': ['parameter_1'],
                'outputs': ['parameter_2'],
                'length': 3,
                'nest': True,
            },
            {
                'name': '',
                'inputs': ['parameter_3'],
                'outputs': ['parameter_4'],
                'length': 6,
                'nest': False,
            },
            {
                'name': '',
                'inputs': ['parameter_2', 'parameter_4'],
                'outputs': ['parameter_5'],
                'length': 2,
            },
        ]
        with self.assertRaises(MissingMergePriority):
            _, _, _ = check_task_compatibility(compat_props)

    def test_incompatible_nesting(self):
        """Ensure raise if nesting is incompatible with task lengths."""
        compat_props = [
            {
                'name': '',
                'inputs': ['parameter_3'],
                'outputs': ['parameter_4'],
                'length': 6,
                'nest': False,
            },
            {
                'name': '',
                'inputs': ['parameter_2', 'parameter_4'],
                'outputs': ['parameter_5'],
                'length': 2,
            },
        ]
        with self.assertRaises(IncompatibleNesting):
            _, _ = check_task_compatibility(compat_props)

    def test_warning_on_unrequired_merge_priority(self):
        """Ensure a warning is issued when `merge_priority` is specified on one or more
        tasks when it is not needed (i.e. nesting is not mixed)."""
        compat_props = [
            {
                'name': '',
                'inputs': ['parameter_1'],
                'outputs': ['parameter_2'],
                'length': 1,
                'nest': True,
                'merge_priority': 0,
            },
        ]
        with self.assertWarns(Warning):
            _, _, _ = check_task_compatibility(compat_props)

    def test_warning_on_unrequired_nest(self):
        """Ensure a warning is issued when `nest` is specified on a task whose output is
        not used by any other task."""
        compat_props = [
            {
                'name': '',
                'inputs': ['parameter_1'],
                'outputs': ['parameter_2'],
                'length': 1,
                'nest': True,
            },
        ]
        with self.assertWarns(Warning):
            _, _, _ = check_task_compatibility(compat_props)

    def test_nest_true_added_by_default(self):
        """Ensure `nest: True` is added by default to tasks where nesting is unspecified
        and whose output is used by another task."""
        compat_props = [
            {
                'name': '',
                'inputs': ['parameter_3'],
                'outputs': ['parameter_4'],
                'length': 6,
            },
            {
                'name': '',
                'inputs': ['parameter_4'],
                'outputs': ['parameter_5'],
                'length': 1,
            },
        ]
        _, compat_props_new, _ = check_task_compatibility(compat_props)
        self.assertTrue(compat_props_new[0].get('nest') == True)


class TaskOrderingTestCase(unittest.TestCase):
    'Tests ensuring correct reordering of tasks.'

    def test_correct_ordering(self):
        compat_props = [
            {
                'name': '',
                'inputs': ['parameter_3'],
                'outputs': ['parameter_4'],
                'length': 1,
            },
            {
                'name': '',
                'inputs': ['parameter_1'],
                'outputs': ['parameter_2'],
                'length': 1,
            },
            {
                'name': '',
                'inputs': ['parameter_1b'],
                'outputs': ['parameter_2b'],
                'length': 1,
            },
            {
                'name': '',
                'inputs': ['parameter_2', 'parameter_2b'],
                'outputs': ['parameter_3'],
                'length': 1,
            },
        ]
        srt_idx, _, _ = check_task_compatibility(compat_props)
        self.assertTrue(
            srt_idx == [1, 2, 3, 0] or srt_idx == [2, 1, 3, 0]
        )
