"""`matflow.api.py`

This module contains the application programming interface (API) to `matflow`,
and includes functions that are called by the command line interface (CLI; in
`matflow.cli.py`).

"""

from pathlib import Path
from pprint import pprint

import pyperclip
from hpcflow import api as hf_api

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
    workflow_dict = parse_workflow_profile(profile_path)

    with profile_path.open('r') as handle:
        profile_str = handle.read()

    workflow = Workflow(**workflow_dict, stage_directory=directory)
    workflow.profile_str = profile_str
    workflow.set_ids()

    if write_dirs:
        workflow.write_directories()
        workflow.write_hpcflow_workflow()
        workflow.save()

        # Copy profile to workflow directory:
        workflow.path.joinpath(profile_path).write_bytes(profile_path.read_bytes())

        # Copy workflow human_id to clipboard, if supported:
        try:
            pyperclip.copy(workflow.human_id)
        except:
            pass

    return workflow


def go(profile_path, directory=None):
    'Generate and submit a new workflow from a profile file.'

    workflow = make_workflow(profile_path, directory=directory, write_dirs=True)
    hf_path = workflow.path.joinpath('1.hf.yml')
    hf_wid = hf_api.make_workflow(dir_path=workflow.path, profile_list=[hf_path])
    hf_api.submit_workflow(workflow_id=hf_wid, dir_path=workflow.path)


def load_workflow(directory, full_path=False):

    path = Path(directory or '').resolve()
    workflow = Workflow.load(path, full_path)

    return workflow


def prepare_task(task_idx, directory):
    'Prepare a task for execution by setting inputs and running input maps.'
    workflow = load_workflow(directory)
    workflow.prepare_task(task_idx)


def process_task(task_idx, directory):
    'Process a completed task by running the output map.'
    workflow = load_workflow(directory)
    workflow.process_task(task_idx)
