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


def make_workflow(profile_path, directory=None):
    """Generate a new Workflow.

    Parameters
    ----------
    profile : str or Path
    directory : str or Path, optional
        The directory in which the Workflow will be generated. By default, this
        is the working (i.e. invoking) directory.    

    Returns
    -------
    Workflow

    """

    profile_path = Path(profile_path)
    stage_dir = Path(directory or '').resolve()
    workflow_dict = parse_workflow_profile(profile_path)

    print('workflow_dict:')
    pprint(workflow_dict)

    workflow = Workflow(**workflow_dict, stage_directory=stage_dir)
    workflow.save_state()

    # Copy profile to workflow directory:
    workflow.path.joinpath(profile_path).write_bytes(profile_path.read_bytes())

    pyperclip.copy(workflow.human_id)

    return workflow


def load_workflow(directory):

    project_dir = Path(directory or '').resolve()
    workflow = Workflow.load_state(project_dir)
    return workflow


def proceed(directory):

    workflow = load_workflow(directory)
    workflow.proceed()
