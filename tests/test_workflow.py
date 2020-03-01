"""Module containing unit tests on Workflow initialisation."""

import unittest
from shutil import rmtree

from matflow import TEST_WORKFLOWS_DIR, TEST_WORKING_DIR
from matflow.api import make_workflow
from matflow.errors import IncompatibleWorkflow
from matflow.models.task import check_task_compatibility


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
                'inputs': ['parameter_1'],
                'length': 1,
                'nest_idx': 0,
                'outputs': ['parameter_2'],
            },
            {
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
                'inputs': ['parameter_1'],
                'length': 1,
                'nest_idx': 0,
                'outputs': ['parameter_2'],
            },
            {
                'inputs': ['parameter_2'],
                'length': 1,
                'nest_idx': 0,
                'outputs': ['parameter_1'],
            },
        ]
        with self.assertRaises(IncompatibleWorkflow):
            check_task_compatibility(task_compat_props)
