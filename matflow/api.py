"""`matflow.api.py`

This module contains the application programming interface (API) to `matflow`,
and includes functions that are called by the command line interface (CLI; in
`matflow.cli.py`).

"""

from pathlib import Path
from pprint import pprint

import pyperclip
from ruamel.yaml import YAML
from hpcflow import kill as hpcflow_kill

from matflow.config import Config
from matflow.extensions import load_extensions
from matflow.profile import parse_workflow_profile
from matflow.models.workflow import Workflow


def make_workflow(profile_path, directory=None, write_dirs=True):
    """Generate a new Workflow from a profile file.

    Parameters
    ----------
    profile : str or Path
        Path to the profile file.
    directory : str or Path, optional
        The directory in which the Workflow will be generated. By default, this
        is the working (i.e. invoking) directory.    

    Returns
    -------
    workflow : Workflow

    """

    load_extensions()

    profile_path = Path(profile_path)
    workflow_dict = parse_workflow_profile(profile_path)

    with profile_path.open('r') as handle:
        profile_str = handle.read()

    workflow = Workflow(**workflow_dict, stage_directory=directory)
    workflow.profile_str = profile_str
    workflow.set_ids()

    if write_dirs:
        workflow.write_directories()
        workflow.dump_hpcflow_workflow_file('hpcflow_workflow.yml')
        workflow.save()

        # Copy profile to workflow directory:
        workflow.path.joinpath(profile_path).write_bytes(profile_path.read_bytes())

        # Copy workflow human_id to clipboard, if supported:
        try:
            pyperclip.copy(workflow.human_id)
        except:
            pass

    return workflow


def submit_workflow(profile_path, directory=None):
    'Generate and submit a new workflow from a profile file.'
    workflow = make_workflow(profile_path, directory=directory, write_dirs=True)
    workflow.submit()


def load_workflow(directory, full_path=False):
    Config.set_config()
    path = Path(directory or '').resolve()
    workflow = Workflow.load(path, full_path)

    return workflow


def prepare_task(task_idx, directory):
    'Prepare a task for execution by setting inputs and running input maps.'
    load_extensions()
    workflow = load_workflow(directory)
    workflow.prepare_task(task_idx)


def process_task(task_idx, directory):
    'Process a completed task by running the output map.'
    load_extensions()
    workflow = load_workflow(directory)
    workflow.process_task(task_idx)


def run_python_task(task_idx, element_idx, directory):
    'Run a (commandless) Python task.'
    load_extensions()
    workflow = load_workflow(directory)
    workflow.run_python_task(task_idx, element_idx)


def append_schema_source(schema_source_path):
    'Add a task schema source file to the end of the schema source list.'
    Config.append_schema_source(schema_source_path)


def prepend_schema_source(schema_source_path):
    'Add a task schema source file to the front of the schema source list.'
    Config.prepend_schema_source(schema_source_path)


def validate():
    load_extensions()


def kill(directory):
    Config.set_config()
    hpcflow_kill(dir_path=directory, config_dir=Config.get('hpcflow_config_dir'))
