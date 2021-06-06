"""`matflow.api.py`

This module contains the application programming interface (API) to `matflow`,
and includes functions that are called by the command line interface (CLI; in
`matflow.cli.py`).

"""

import copy
from pathlib import Path

import pyperclip
from hpcflow import kill as hpcflow_kill
from hpcflow import cloud_connect as hpcflow_cloud_connect

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

    profile = {'file': profile_str, 'parsed': copy.deepcopy(workflow_dict)}

    iterate_run_opts = {
        **Config.get('default_sticky_iterate_run_options'),
        **Config.get('default_iterate_run_options'),
    }
    workflow_dict.update({'iterate_run_options': iterate_run_opts})

    workflow = Workflow(**workflow_dict, stage_directory=directory, profile=profile)
    workflow.set_ids()

    if write_dirs:
        workflow.write_HDF5_file()
        workflow.write_directories()
        workflow.prepare_iteration(iteration_idx=0)
        workflow.dump_hpcflow_workflow_file('hpcflow_workflow.yml')

        # Copy profile to workflow directory:
        workflow.path.joinpath(profile_path.name).write_bytes(profile_path.read_bytes())

        # Copy workflow human_id to clipboard, if supported:
        try:
            pyperclip.copy(workflow.human_id)
        except:
            pass

    return workflow


def submit_workflow(workflow_path, directory=None):
    """Generate and submit a new workflow from a profile file.    

    Parameters
    ----------
    workflow_path : str or Path
        Path to either a profile file or a workflow project directory that contains a 
        previously generated workflow HDF5 file.
    directory : str or Path, optional
        Applicable if `workflow_path` points to a profile file. The directory in which the
        Workflow will be generated. By default, this is the working (i.e. invoking)
        directory.

    Returns
    -------
    None

    """

    if Path(workflow_path).is_file():
        workflow = make_workflow(workflow_path, directory=directory, write_dirs=True)
    else:
        load_extensions()
        workflow = load_workflow(workflow_path)

    workflow.submit()


def load_workflow(directory, full_path=False):
    Config.set_config()
    path = Path(directory or '').resolve()
    workflow = Workflow.load_HDF5_file(path, full_path)

    return workflow


def prepare_task(task_idx, iteration_idx, directory, is_array=False):
    """Prepare a task (iteration) for execution by setting inputs and running input
    maps."""

    load_extensions()
    workflow = load_workflow(directory)
    workflow.prepare_task(task_idx, iteration_idx, is_array=is_array)


def prepare_task_element(task_idx, element_idx, directory, is_array=False):
    """Prepare a task element for execution by setting inputs and running input maps."""
    load_extensions()
    workflow = load_workflow(directory)
    workflow.prepare_task_element(task_idx, element_idx, is_array=is_array)


def process_task(task_idx, iteration_idx, directory, is_array=False):
    """Process a completed task (iteration) by running the output map."""
    load_extensions()
    workflow = load_workflow(directory)
    workflow.process_task(task_idx, iteration_idx, is_array=is_array)


def process_task_element(task_idx, element_idx, directory, is_array=False):
    """Process a task element for execution by running output maps and saving outputs."""
    load_extensions()
    workflow = load_workflow(directory)
    workflow.process_task_element(task_idx, element_idx, is_array=is_array)


def run_python_task(task_idx, element_idx, directory):
    """Run a (commandless) Python task."""
    load_extensions()
    workflow = load_workflow(directory)
    workflow.run_python_task(task_idx, element_idx)


def prepare_sources(task_idx, iteration_idx, directory):
    """Prepare source files."""
    load_extensions()
    workflow = load_workflow(directory)
    workflow.prepare_sources(task_idx, iteration_idx)


def append_schema_source(schema_source_path):
    """Add a task schema source file to the end of the schema source list."""
    Config.append_schema_source(schema_source_path)


def prepend_schema_source(schema_source_path):
    """Add a task schema source file to the front of the schema source list."""
    Config.prepend_schema_source(schema_source_path)


def validate():
    load_extensions()


def kill(directory):
    Config.set_config()
    hpcflow_kill(dir_path=directory, config_dir=Config.get('hpcflow_config_dir'))


def cloud_connect(provider):
    Config.set_config()
    hpcflow_cloud_connect(provider, config_dir=Config.get('hpcflow_config_dir'))


def write_element_directories(iteration_idx, directory):
    'Generate element directories for a given iteration.'
    load_extensions()
    workflow = load_workflow(directory)
    if workflow.iterate:
        num_iters = workflow.iterate['num_iterations']
    else:
        num_iters = workflow.num_iterations
    if iteration_idx < num_iters:
        workflow.write_element_directories(iteration_idx)
        workflow.prepare_iteration(iteration_idx)


def archive(directory, archive):
    """Perform an on-demand archive of an existing workflow."""
    workflow = load_workflow(directory)
    workflow.do_archive(archive)


def get_task_schemas():
    Config.set_config()
    return Config.get('task_schemas')
