"""Module containing unit tests on Workflow initialisation."""

import unittest
from shutil import rmtree

from matflow import TEST_WORKFLOWS_DIR, TEST_WORKING_DIR
from matflow.api import make_workflow
from matflow.models.workflow import IncompatibleWorkflow


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
        path = TEST_WORKFLOWS_DIR.joinpath('test_output_non_exclusivity.yml')
        with self.assertRaises(IncompatibleWorkflow):
            make_workflow(path, directory=TEST_WORKING_DIR)

    def test_circular_reference(self):
        """Ensure raises on a workflow whose Tasks are circularly referential."""
        path = TEST_WORKFLOWS_DIR.joinpath('test_circular_reference.yml')
        with self.assertRaises(IncompatibleWorkflow):
            make_workflow(path, directory=TEST_WORKING_DIR)
