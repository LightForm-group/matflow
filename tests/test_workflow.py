"""Module containing unit tests on Workflow initialisation."""

import unittest

from matflow.errors import IncompatibleWorkflow
from matflow.models import TaskSchema
from matflow.models.construction import get_dependency_idx

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
    """Construct TaskSchema objects for TaskDependencyTestCase tests."""
    for idx, i in enumerate(task_lst):
        task_lst[idx]['schema'] = TaskSchema(**i['schema'])
    return task_lst


class TaskDependencyTestCase(unittest.TestCase):
    """Tests on `get_dependency_idx`"""

    def test_single_dependency(self):
        """Test correct dependency index for a single task dependency."""
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
        """Test single dependencies for two parallel contexts."""
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
        """Test where a task depends on two tasks."""
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
        """Test raises on multiple tasks that include the same output (and context)."""
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
        """Test raises on circularly referential Tasks."""
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
