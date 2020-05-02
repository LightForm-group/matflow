"""`matflow.api.py`

This module contains the application programming interface (API) to `matflow`,
and includes functions that are called by the command line interface (CLI; in
`matflow.cli.py`).

"""

from pathlib import Path
from pprint import pprint

import pyperclip

from matflow.profile import parse_workflow_profile
from matflow.models import Workflow


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

    profile_path = Path(profile_path)
    stage_dir = Path(directory or '').resolve()
    workflow_dict = parse_workflow_profile(profile_path)

    with profile_path.open('r') as handle:
        profile_str = handle.read()

    workflow = Workflow(**workflow_dict, stage_directory=stage_dir)

    workflow.profile_str = profile_str
    workflow.set_ids()

    if write_dirs:
        workflow.write_directories()
        workflow.write_hpcflow_workflow()
        workflow.save_state()

        # Copy profile to workflow directory:
        workflow.path.joinpath(profile_path).write_bytes(profile_path.read_bytes())

        # Copy workflow human_id to clipboard, if supported:
        try:
            pyperclip.copy(workflow.human_id)
        except:
            pass

    return workflow


def load_workflow(directory, full_path=False):

    path = Path(directory or '').resolve()
    workflow = Workflow.load_state(path, full_path)

    return workflow


def proceed(directory):

    workflow = load_workflow(directory)
    workflow.proceed()


def prepare_task(task_idx, directory):
    'Prepare a task for execution by setting inputs and running input maps.'
    workflow = load_workflow(directory)
    workflow.prepare_task(task_idx)


def process_task(task_idx, directory):
    'Process a completed task by running the output map.'
    workflow = load_workflow(directory)
    workflow.process_task(task_idx)
