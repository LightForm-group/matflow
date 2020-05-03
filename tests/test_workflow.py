"""Module containing unit tests on Workflow initialisation."""

import unittest

from matflow.errors import (IncompatibleWorkflow, IncompatibleTaskNesting,
                            MissingMergePriority)
from matflow.models.task import TaskSchema
from matflow.models.workflow import get_dependency_idx

"""
tests for inputs/outputs_idx:
- for a variety of scenarios, check all parameters from the same task have the same number of elements_idx.
- for a few scenarios, check expected elements_idx and task_idx.
- check all keys of output (i.e. `task_idx`) are exactly the set of task_idx values in downstream + upstream tasks.
- check works when no upstream tasks.

tests for resolve_task_num_elements:
- check works when no upstream tasks

"""


def init_schemas(task_lst):
    'Construct TaskSchema objects for TaskDependencyTestCase tests.'
    for idx, i in enumerate(task_lst):
        task_lst[idx]['schema'] = TaskSchema(**i['schema'])
    return task_lst


class TaskDependencyTestCase(unittest.TestCase):
    'Tests on `get_dependency_idx`'

    def test_single_dependency(self):
        'Test correct dependency index for a single task dependency.'
        task_lst = [
            {
                'context': '',
                'schema': {
                    'name': 'one',
                    'inputs': [
                        {'name': 'p1', 'context': None},
                        {'name': 'p2', 'context': None},
                    ],
                    'outputs': ['p3'],
                },
            },
            {
                'context': '',
                'schema': {
                    'name': 'one',
                    'inputs': [
                        {'name': 'p3', 'context': None},
                        {'name': 'p4', 'context': None},
                    ],
                    'outputs': ['p5'],
                },
            },
        ]
        dep_idx = get_dependency_idx(init_schemas(task_lst))
        dep_idx_exp = [[], [0]]
        self.assertTrue(dep_idx == dep_idx_exp)

    def test_single_dependency_two_contexts(self):
        'Test single dependencies for two parallel contexts.'
        task_lst = [
            {
                'context': 'context_A',
                'schema': {
                    'name': 'one',
                    'inputs': [
                        {'name': 'p1', 'context': None},
                        {'name': 'p2', 'context': None},
                    ],
                    'outputs': ['p3'],
                },
            },
            {
                'context': 'context_A',
                'schema': {
                    'name': 'one',
                    'inputs': [
                        {'name': 'p3', 'context': None},
                        {'name': 'p4', 'context': None},
                    ],
                    'outputs': ['p5'],
                },
            },
            {
                'context': 'context_B',
                'schema': {
                    'name': 'one',
                    'inputs': [
                        {'name': 'p1', 'context': None},
                        {'name': 'p2', 'context': None},
                    ],
                    'outputs': ['p3'],
                },
            },
            {
                'context': 'context_B',
                'schema': {
                    'name': 'one',
                    'inputs': [
                        {'name': 'p3', 'context': None},
                        {'name': 'p4', 'context': None},
                    ],
                    'outputs': ['p5'],
                },
            },
        ]
        dep_idx = get_dependency_idx(init_schemas(task_lst))
        dep_idx_exp = [[], [0], [], [2]]
        self.assertTrue(dep_idx == dep_idx_exp)

    def test_two_dependencies(self):
        'Test where a task depends on two tasks.'
        task_lst = [
            {
                'context': 'contextA',
                'schema': {
                    'name': 'one',
                    'inputs': [
                        {'name': 'p1', 'context': None},
                        {'name': 'p2', 'context': None},
                    ],
                    'outputs': ['p3', 'p4'],
                },
            },
            {
                'context': 'contextB',
                'schema': {
                    'name': 'one',
                    'inputs': [
                        {'name': 'p1', 'context': None},
                        {'name': 'p2', 'context': None},
                    ],
                    'outputs': ['p3', 'p4'],
                },
            },
            {
                'context': '',
                'schema': {
                    'name': 'one',
                    'inputs': [
                        {'name': 'p3', 'context': 'contextA'},
                        {'name': 'p4', 'context': 'contextB'},
                    ],
                    'outputs': ['p5'],
                },
            },
        ]
        dep_idx = get_dependency_idx(init_schemas(task_lst))
        dep_idx_exp = [[], [], [0, 1]]
        self.assertTrue(dep_idx == dep_idx_exp)

    def test_raise_on_output_non_exclusivity(self):
        'Test raises on multiple tasks that include the same output (and context).'
        task_lst = [
            {
                'context': '',
                'schema': {
                    'name': 'one',
                    'inputs': [
                        {'name': 'p1', 'context': None},
                        {'name': 'p2', 'context': None},
                    ],
                    'outputs': ['p3'],
                },
            },
            {
                'context': '',
                'schema': {
                    'name': 'two',
                    'inputs': [
                        {'name': 'p4', 'context': None},
                    ],
                    'outputs': ['p3'],
                },
            },
        ]
        with self.assertRaises(IncompatibleWorkflow):
            get_dependency_idx(init_schemas(task_lst))

    def test_raise_on_circular_reference(self):
        'Test raises on circularly referential Tasks.'
        task_lst = [
            {
                'context': '',
                'schema': {
                    'name': 'one',
                    'inputs': [
                        {'name': 'p1', 'context': None},
                    ],
                    'outputs': ['p2'],
                },
            },
            {
                'context': '',
                'schema': {
                    'name': 'two',
                    'inputs': [
                        {'name': 'p2', 'context': None},
                    ],
                    'outputs': ['p1'],
                },
            },
        ]
        with self.assertRaises(IncompatibleWorkflow):
            get_dependency_idx(init_schemas(task_lst))


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
        with self.assertRaises(IncompatibleTaskNesting):
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
