"""matflow.models.workflow.py

Module containing the Workflow class and some functions used to decorate Workflow methods.

"""

import copy
import functools
import secrets
import pickle
from datetime import datetime
from enum import Enum
from pathlib import Path
from pprint import pprint
from subprocess import run, PIPE
from warnings import warn

import hickle
import hpcflow
import numpy as np
from ruamel.yaml import YAML

from matflow import __version__
from matflow.config import Config
from matflow.errors import (
    IncompatibleTaskNesting,
    MissingMergePriority,
    WorkflowPersistenceError,
    TaskElementExecutionError,
)
from matflow.hicklable import to_hicklable
from matflow.utils import parse_times, zeropad, datetime_to_dict
from matflow.models.command import DEFAULT_FORMATTERS
from matflow.models.construction import init_tasks


def requires_ids(func):
    'Workflow method decorator to raise if IDs are not assigned.'
    @functools.wraps(func)
    def func_wrap(self, *args, **kwargs):
        if not self.id:
            raise ValueError('Run `set_ids()` before using this method.')
        return func(self, *args, **kwargs)
    return func_wrap


def requires_path_exists(func):
    'Workflow method decorator to raise if workflow path does not exist as a directory.'
    @functools.wraps(func)
    def func_wrap(self, *args, **kwargs):
        if not self.path_exists:
            raise ValueError('Run `write_directories` before using this method.')
        return func(self, *args, **kwargs)
    return func_wrap


class WorkflowAction(Enum):

    generate = 1
    submit = 2
    prepare_task = 3
    process_task = 4


class Workflow(object):

    __slots__ = [
        '_id',
        '_human_id',
        '_profile_str',
        '_is_from_file',
        '_name',
        '_extend_paths',
        '_extend_nest_idx',
        '_stage_directory',
        '_tasks',
        '_elements_idx',
        '_history',
    ]

    def __init__(self, name, tasks, stage_directory=None, extend=None,
                 check_integrity=True, __is_from_file=False):

        self._id = None             # Assigned once by set_ids()
        self._human_id = None       # Assigned once by set_ids()
        self._profile_str = None    # Assigned once in `profile_str` setter

        self._is_from_file = __is_from_file
        self._name = name
        self._extend_paths = [str(Path(i).resolve())
                              for i in extend['paths']] if extend else None
        self._extend_nest_idx = extend['nest_idx'] if extend else None
        self._stage_directory = str(Path(stage_directory or '').resolve())

        tasks, elements_idx = init_tasks(tasks, self.is_from_file, check_integrity)
        self._tasks = tasks
        self._elements_idx = elements_idx

        if not self.is_from_file:
            self._history = []
            self._append_history(WorkflowAction.generate)

    def set_ids(self):
        if self._id:
            raise ValueError(f'IDs are already set for workflow. ID is: "{self.id}"; '
                             f'human ID is "{self.human_id}".')
        else:
            self._human_id = self.name_safe + '_' + parse_times('%Y-%m-%d-%H%M%S')[0]
            self._id = secrets.token_hex(15)

    def __len__(self):
        return len(self.tasks)

    def _append_history(self, action, **kwargs):
        'Append a new history event.'

        if action not in WorkflowAction:
            raise TypeError('`action` must be a `WorkflowAction`.')

        new = {
            'action': action,
            'matflow_version': __version__,
            'timestamp': datetime.now(),
            'action_info': kwargs,
        }
        self._history.append(new)
        if action is not WorkflowAction.generate:
            self.save()

    @property
    def version(self):
        return len(self._history)

    @property
    def history(self):
        return tuple(self._history)

    @property
    def id(self):
        return self._id

    @property
    def human_id(self):
        return self._human_id

    @property
    def is_from_file(self):
        return self._is_from_file

    @property
    def name(self):
        return self._name

    @property
    def name_safe(self):
        'Get name without spaces'
        return self.name.replace(' ', '_')

    @property
    def profile_str(self):
        'Get, as a string, the profile file that was used to construct this workflow.'
        return self._profile_str

    @profile_str.setter
    def profile_str(self, profile_str):
        if self._profile_str:
            raise ValueError(f'`profile_str` is already set for the workflow')
        else:
            self._profile_str = profile_str

    @property
    def tasks(self):
        return self._tasks

    @property
    def elements_idx(self):
        return self._elements_idx

    @property
    def extend_paths(self):
        if self._extend_paths:
            return [Path(i) for i in self._extend_paths]
        else:
            return None

    @property
    def extend_nest_idx(self):
        return self._extend_nest_idx

    @property
    def stage_directory(self):
        return Path(self._stage_directory)

    @property
    def path_exists(self):
        'Does the Workflow project directory exist on this machine?'
        try:
            path = self.path
        except ValueError:
            return False
        if path.is_dir():
            return True
        else:
            return False

    @property
    @requires_ids
    def path(self):
        'Get the full path of the Workflow project as a Path.'
        return Path(self.stage_directory, self.human_id)

    @property
    def path_str(self):
        'Get the full path of the Workflow project as a string.'
        return str(self.path)

    @property
    def hdf5_path(self):
        return self.path.joinpath(f'workflow_v{self.version:03}.hdf5')

    def get_task_idx_padded(self, task_idx):
        'Get a task index, zero-padded according to the number of tasks.'
        return zeropad(task_idx, len(self) - 1)

    @requires_path_exists
    def get_task_path(self, task_idx):
        'Get the path to a task directory.'
        if task_idx > (len(self) - 1):
            msg = f'Workflow has only {len(self)} tasks.'
            raise ValueError(msg)
        task = self.tasks[task_idx]
        task_idx_fmt = self.get_task_idx_padded(task_idx)
        task_path = self.path.joinpath(f'task_{task_idx_fmt}_{task.name}')
        return task_path

    @requires_path_exists
    def get_element_path(self, task_idx, element_idx):
        'Get the path to an element directory.'
        num_elements = self.elements_idx[task_idx]['num_elements']
        if element_idx > (num_elements - 1):
            msg = f'Task at index {task_idx} has only {num_elements} elements.'
            raise ValueError(msg)

        element_path = self.get_task_path(task_idx)
        if num_elements > 1:
            element_idx_fmt = str(zeropad(element_idx, num_elements - 1))
            element_path = element_path.joinpath(element_idx_fmt)

        return element_path

    @requires_path_exists
    def _get_element_temp_output_path(self, task_idx, element_idx):
        task = self.tasks[task_idx]
        element_path = self.get_element_path(task_idx, element_idx)
        out = element_path.joinpath(f'task_output_{task.id}_element_{element_idx}.pickle')
        return out

    def write_directories(self):
        'Generate task and element directories.'

        if self.path.exists():
            raise ValueError('Directories for this workflow already exist.')

        self.path.mkdir(exist_ok=False)

        for elems_idx, task in zip(self.elements_idx, self.tasks):

            # Generate task directory:
            self.get_task_path(task.task_idx).mkdir()

            num_elems = elems_idx['num_elements']
            # Generate element directories:
            for i in range(num_elems):
                self.get_element_path(task.task_idx, i).mkdir(exist_ok=True)

    @requires_path_exists
    def get_hpcflow_workflow(self):
        'Generate an hpcflow workflow to execute this workflow.'

        command_groups = []
        variables = {}
        for elems_idx, task in zip(self.elements_idx, self.tasks):

            if task.schema.is_func:
                # The task is to be run directly in Python:
                # (SGE specific)
                fmt_commands = [
                    f'matflow run-python-task --task-idx={task.task_idx} '
                    f'--element-idx=$(($SGE_TASK_ID-1)) '
                    f'--directory={self.path}'
                ]

            else:
                # `input_vars` are those inputs that appear directly in the commands:
                cmd_group = task.schema.command_group
                fmt_commands, input_vars = cmd_group.get_formatted_commands(
                    task.local_inputs['inputs'].keys())

                cmd_line_inputs = {}
                for local_in_name, local_in in task.local_inputs['inputs'].items():
                    if local_in_name in input_vars:
                        # TODO: We currently only consider input_vars for local inputs.

                        # Expand values for intra-task nesting:
                        values = [local_in['vals'][i] for i in local_in['vals_idx']]

                        # Format values:
                        fmt_func_scope = Config.get('CLI_arg_maps').get((
                            task.schema.name,
                            task.schema.method,
                            task.schema.implementation
                        ))
                        fmt_func = None
                        if fmt_func_scope:
                            fmt_func = fmt_func_scope.get(local_in_name)
                        if not fmt_func:
                            fmt_func = DEFAULT_FORMATTERS.get(
                                type(values[0]),
                                lambda x: str(x)
                            )

                        values_fmt = [fmt_func(i) for i in values]

                        # Expand values for inter-task nesting:
                        values_fmt_all = [
                            values_fmt[i]
                            for i in elems_idx['inputs'][local_in_name]['input_idx']
                        ]
                        cmd_line_inputs.update({local_in_name: values_fmt_all})

                for local_in_name, var_name in input_vars.items():

                    var_file_name = '{}.txt'.format(var_name)
                    variables.update({
                        var_name: {
                            'file_contents': {
                                'path': var_file_name,
                                'expected_multiplicity': 1,
                            },
                            'value': '{}',
                        }
                    })

                    # Create text file in each element directory for each in `input_vars`:
                    for i in range(elems_idx['num_elements']):

                        task_elem_path = self.get_element_path(task.task_idx, i)
                        in_val = cmd_line_inputs[local_in_name][i]

                        var_file_path = task_elem_path.joinpath(var_file_name)
                        with var_file_path.open('w') as handle:
                            handle.write(in_val + '\n')

            scheduler_opts = {}
            scheduler_opts_process = Config.get('prepare_process_scheduler_options')
            for k, v in task.run_options.items():
                if k != 'num_cores':
                    if k == 'pe':
                        v = v + ' ' + str(task.run_options['num_cores'])
                    scheduler_opts.update({k: v})

            task_path_rel = str(self.get_task_path(task.task_idx).name)

            environment = task.software_instance.get('environment', [])
            task_idx_fmt = self.get_task_idx_padded(task.task_idx)
            command_groups.extend([
                {
                    'directory': '.',
                    'nesting': 'hold',
                    'commands': [
                        'matflow prepare-task --task-idx={}'.format(task.task_idx)
                    ],
                    'environment': environment,
                    'stats': False,
                    'scheduler_options': scheduler_opts_process,
                    'name': f't{task_idx_fmt}_pre',
                },
                {
                    'directory': '<<{}_dirs>>'.format(task_path_rel),
                    'nesting': 'hold',
                    'commands': fmt_commands,
                    'environment': environment,
                    'stats': task.stats,
                    'scheduler_options': scheduler_opts,
                    'name': f't{task_idx_fmt}',
                },
                {
                    'directory': '.',
                    'nesting': 'hold',
                    'commands': [
                        'matflow process-task --task-idx={}'.format(task.task_idx)
                    ],
                    'stats': False,
                    'scheduler_options': scheduler_opts_process,
                    'name': f't{task_idx_fmt}_post',
                },
            ])

            # Add variable for the task directories:
            elem_dir_regex = '/[0-9]+$' if elems_idx['num_elements'] > 1 else ''
            variables.update({
                '{}_dirs'.format(task_path_rel): {
                    'file_regex': {
                        'pattern': f'({task_path_rel}{elem_dir_regex})',
                        'is_dir': True,
                        'group': 0,
                    },
                    'value': '{}',
                }
            })

        hf_data = {
            'scheduler': 'sge',
            'output_dir': 'output',
            'error_dir': 'output',
            'command_groups': command_groups,
            'variables': variables,
        }

        return hf_data

    @requires_path_exists
    def dump_hpcflow_workflow_file(self, file_name):

        dt_stamp = datetime.now().strftime(r'%Y.%m.%d at %H:%M:%S')
        about_msg = (
            f'# --- hpcflow profile generated by matflow (version: '
            f'{__version__}) on {dt_stamp} ---\n'
            f'# Note: modifications to this profile do NOT modify the matflow workflow '
            f'submission.\n'
            f'\n'
        )
        hf_data = self.get_hpcflow_workflow()
        with self.path.joinpath(file_name).open('w') as handle:
            handle.write(about_msg)
            yaml = YAML()
            yaml.width = 5000  # Avoid line breaks
            yaml.indent(mapping=2, sequence=4, offset=2)
            yaml.dump(hf_data, handle)

    @requires_path_exists
    def submit(self):
        hf_data = self.get_hpcflow_workflow()
        hf_wid = hpcflow.make_workflow(
            dir_path=self.path,
            workflow_dict=hf_data,
            config_dir=Config.get('hpcflow_config_dir'),
        )
        self._append_history(WorkflowAction.submit, hpcflow_version=hpcflow.__version__)
        hpcflow.submit_workflow(
            workflow_id=hf_wid,
            dir_path=self.path,
            config_dir=Config.get('hpcflow_config_dir'),
        )

    def get_extended_workflows(self):
        if self.extend_paths:
            return [Workflow.load(i, full_path=True) for i in self.extend_paths]
        else:
            return None

    def as_dict(self):
        'Return attributes dict with preceding underscores removed.'
        out = {k.lstrip('_'): getattr(self, k) for k in self.__slots__}

        # Deal with the WorkflowAction enums and datetimes in history action values:
        history = []
        for i in out['history']:
            hist = copy.deepcopy(i)
            hist['action'] = (hist['action'].name, hist['action'].value)
            hist['timestamp'] = datetime_to_dict(hist['timestamp'])
            history.append(hist)
        out['history'] = history

        return out

    @classmethod
    def get_existing_workflow_files(cls, directory):
        """Get the ID and versions of any workflow files within a directory.

        Parameters
        ----------
        directory : str or Path
            Directory in which to search for workflow files.

        Returns
        -------
        existing_files : dict of dict
            A dict whose keys are the full file paths (Path objects) to any parseable
            workflow files, and whose keys are dicts with keys: "id" and "version".

        """

        directory = Path(directory)
        if not directory.is_dir():
            raise TypeError('`directory` is not an existing directory!')

        existing_files = {}
        for i in directory.glob('*'):
            if i.is_file():
                try:
                    with i.open() as handle:
                        hickle.load(handle)
                except OSError:
                    continue
                try:
                    wkflow = cls.load(i, full_path=True)
                    id_, version = wkflow.id, wkflow.version
                except:
                    continue

                existing_files.update({i: {'id': id_, 'version': version}})

        return existing_files

    @requires_path_exists
    def save(self, path=None, keep_previous_versions=False):
        """Save workflow to an HDF5 file.

        Parameters
        ----------
        path : str or Path, optional
            If specified, must be the full path (including file name) where the workflow
            file should be saved. By default, `None`, in which case the `hdf5_path`
            attribute will be used as the full path.
        keep_previous_versions : bool, optional
            If False, all workflow files with the same ID and lower (or equal) version
            numbers in the same directory as the save location will be deleted. By
            default, False. If True, no existing workflow files in the save directory
            will be deleted.

        Notes
        -----
        - A warning is issued if an existing workflow file exists with the same ID and
          version. The file will be removed if `keep_previous_versions=False`.

        Raises
        ------
        WorkflowPersistenceError 
            If saving was not successful.

        """

        path = Path(path or self.hdf5_path)
        save_dir = path.parent

        if not keep_previous_versions:
            remove_tag = f'.remove_{secrets.token_hex(8)}'

            same_IDs = {k: v
                        for k, v in self.get_existing_workflow_files(save_dir).items()
                        if v['id'] == self.id}

            to_delete = {k: v for k, v in same_IDs.items()
                         if v['version'] <= self.version}

            if self.version in [v['version'] for k, v in to_delete.items()]:
                warn('A saved workflow with the same ID and version already exists in '
                     'this directory. This will be removed.')

            # Mark older versions for deletion:
            for del_path_i in to_delete.keys():
                tagged_name = del_path_i.with_name(del_path_i.name + remove_tag)
                del_path_i.rename(tagged_name)
                msg = f'Marking old workflow file for deletion: {del_path_i}'
                print(msg, flush=True)

        if path.exists():
            msg = f'Workflow cannot be saved to a path that already exists: "{path}".'
            raise WorkflowPersistenceError(msg)

        workflow_as_dict = self.as_dict()
        del workflow_as_dict['is_from_file']
        workflow_as_dict['tasks'] = [i.as_dict() for i in self.tasks]

        err_msg = None
        try:
            obj_json = to_hicklable(workflow_as_dict)
            try:
                with path.open('w') as handle:
                    hickle.dump(obj_json, handle)
            except Exception as err:
                err_msg = f'Failed to save workflow to path: "{path}": {err}.'
        except Exception as err:
            err_msg = (f'Failed to convert Workflow object to `hickle`-compatible '
                       f'dict: {err}.')

        if err_msg:
            if not keep_previous_versions:
                # Revert older versions back to original file names (don't delete):
                for del_path_i in to_delete.keys():
                    tagged_name = del_path_i.with_name(del_path_i.name + remove_tag)
                    tagged_name.rename(del_path_i)
                    msg = f'Save failed. Reverting old workflow file name: {tagged_name}.'
                    print(msg, flush=True)

                del to_delete
            raise WorkflowPersistenceError(err_msg)

        else:
            if not keep_previous_versions:
                # Delete older versions (same ID):
                for del_path_i in to_delete.keys():
                    tagged_name = del_path_i.with_name(del_path_i.name + remove_tag)
                    print(f'Removing old workflow file: {tagged_name}', flush=True)
                    tagged_name.unlink()

    @classmethod
    def load(cls, path, full_path=False, version=None, check_integrity=True):
        """Load workflow from an HDF5 file.

        Parameters
        ----------
        path : str or Path
            Either the directory in which to search for a suitable workflow file (if
            `full_path=False`), or the full path to a workflow file (if `full_path=True`).
            If multiple workflow files with distinct IDs exist in the loading directory,
            an exception is raised. If multiple versions exist (with the same ID), the 
            workflow with the largest version number is loaded, by default, unless
            `version` is specified.
        full_path : bool, optional
            Determines whether `path` is a full workflow file path or a directory path.
            By default, False.
        version : int, optional
            Has effect if `full_path=False`. If specified, a workflow with the specified
            version will be loaded, if it exists, otherwise an exception will be raised.
            Not specified by default.
        check_integrity : bool, optional
            If True, do some checks that the loaded information makes sense. True by
            default.

        Returns
        -------
        workflow : Workflow

        """

        path = Path(path)

        if full_path:
            if not path.is_file():
                raise OSError(f'Workflow file does not exist: "{path}".')
        else:
            existing = cls.get_existing_workflow_files(path)
            if not existing:
                raise ValueError('No workflow files found.')
            all_IDs = set([v['id'] for v in existing.values()])
            if len(all_IDs) > 1:
                msg = (f'Saved workflows with multiple distinct IDs exist in the loading'
                       f' directory "{path}". Specify `path` as the full path to the '
                       f'workflow file, and set `full_path=True`.')
                raise WorkflowPersistenceError(msg)
            else:
                if version:
                    if not isinstance(version, int):
                        raise TypeError('Specify `version` as an integer.')
                    # Load this specific version:
                    paths = [k for k, v in existing.items() if v['version'] == version]
                    if not paths:
                        msg = (f'Workflow with version number "{version}" not found in '
                               f'the loading directory: "{path}".')
                        raise WorkflowPersistenceError(msg)
                    else:
                        path = paths[0]
                else:
                    # Get full path of workflow file with the largest version number:
                    path = sorted(existing.items(), key=lambda i: i[1]['version'])[-1][0]

        try:
            with path.open() as handle:
                obj_json = hickle.load(handle)
        except Exception as err:
            msg = f'Could not load workflow file with `hickle`: "{path}": {err}.'
            raise WorkflowPersistenceError(msg)

        extend = None
        if obj_json['extend_paths']:
            extend = {
                'paths': obj_json['extend_paths'],
                'nest_idx': obj_json['extend_nest_idx']
            }

        obj = {
            'name': obj_json['name'],
            'tasks': obj_json['tasks'],
            'stage_directory': obj_json['stage_directory'],
            'extend': extend,
        }

        workflow = cls(
            **obj,
            _Workflow__is_from_file=True,
            check_integrity=check_integrity,
        )

        workflow.profile_str = obj_json['profile_str']
        workflow._human_id = obj_json['human_id']
        workflow._id = obj_json['id']

        for i in obj_json['history']:
            i['action'] = WorkflowAction(i['action'][1])
            i['timestamp'] = datetime(**i['timestamp'])
        workflow._history = obj_json['history']

        return workflow

    @requires_path_exists
    def prepare_task(self, task_idx):
        'Prepare inputs and run input maps.'

        task = self.tasks[task_idx]
        elems_idx = self.elements_idx[task_idx]
        num_elems = elems_idx['num_elements']
        inputs = [{} for _ in range(num_elems)]
        files = [{} for _ in range(num_elems)]

        # Populate task inputs:
        for input_alias, inputs_idx in elems_idx['inputs'].items():

            task_idx = inputs_idx.get('task_idx')
            input_name = [i['name'] for i in task.schema.inputs
                          if i['alias'] == input_alias][0]

            if task_idx is not None:
                # Input values should be copied from a previous task's `outputs`
                prev_task = self.tasks[task_idx]
                prev_outs = prev_task.outputs

                if not prev_outs:
                    msg = ('Task "{}" does not have the outputs required to parametrise '
                           'the current task: "{}".')
                    raise ValueError(msg.format(prev_task.name, task.name))

                values_all = [[prev_outs[j][input_name] for j in i]
                              for i in inputs_idx['element_idx']]

            else:
                # Input values should be copied from this task's `local_inputs`

                # Expand values for intra-task nesting:
                local_in = task.local_inputs['inputs'][input_name]
                values = [local_in['vals'][i] for i in local_in['vals_idx']]

                # Expand values for inter-task nesting:
                values_all = [values[i] for i in inputs_idx['input_idx']]

            for element, val in zip(inputs, values_all):
                if (task_idx is not None) and (inputs_idx['group'] == 'default'):
                    val = val[0]
                element.update({input_alias: val})

        task.inputs = inputs

        # Run any input maps:
        schema_id = (task.name, task.method, task.software)
        in_map_lookup = Config.get('input_maps').get(schema_id)
        for elem_idx, elem_inputs in zip(range(num_elems), task.inputs):

            task_elem_path = self.get_element_path(task.task_idx, elem_idx)

            # For each input file to be written, invoke the function:
            for in_map in task.schema.input_map:

                # Filter only those inputs required for this file:
                in_map_inputs = {
                    key: val for key, val in elem_inputs.items()
                    if key in in_map['inputs']
                }
                file_path = task_elem_path.joinpath(in_map['file'])

                # TODO: check file_path exists, unit test this as well.

                # Run input map to generate required input files:
                func = in_map_lookup[in_map['file']]
                func(path=file_path, **in_map_inputs)

                # Save generated file as string in workflow:
                with file_path.open('r') as handle:
                    files[elem_idx].update({in_map['file']: handle.read()})

        task.files = files

        # Get software versions:
        software_versions_func = Config.get('software_versions')[task.software]
        software_versions = software_versions_func()
        self._append_history(
            WorkflowAction.prepare_task,
            software_versions=software_versions,
        )

    @requires_path_exists
    def run_python_task(self, task_idx, element_idx):
        'Execute a task that is to be run directly in Python (via the function mapper).'

        task = self.tasks[task_idx]
        schema_id = (task.name, task.method, task.software)
        func = Config.get('func_maps')[schema_id]
        inputs = task.inputs[element_idx]
        try:
            outputs = func(**inputs)
        except Exception as err:
            msg = (f'Task function "{func.__name__}" from module "{func.__module__}" '
                   f'in extension "{func.__module__.split(".")[0]}" has failed with '
                   f'exception: {err}')
            raise TaskElementExecutionError(msg)

        # Save outputs to a temporary pickle file in the element directory. Once all
        # elements have run, we can collect the outputs in `process_task` and save them
        # into the workflow file.
        outputs_path = self._get_element_temp_output_path(task_idx, element_idx)
        with outputs_path.open('wb') as handle:
            pickle.dump(outputs, handle)

    @requires_path_exists
    def process_task(self, task_idx):
        'Process outputs from an executed task: run output map and save outputs.'

        task = self.tasks[task_idx]
        elems_idx = self.elements_idx[task_idx]
        num_elems = elems_idx['num_elements']
        outputs = [{} for _ in range(num_elems)]

        schema_id = (task.name, task.method, task.software)
        out_map_lookup = Config.get('output_maps').get(schema_id)

        # Save hpcflow task stats
        hf_stats_all = hpcflow.get_stats(
            self.path,
            jsonable=True,
            datetime_dicts=True,
            config_dir=Config.get('hpcflow_config_dir'),
        )

        workflow_idx = 0
        submission_idx = 0
        hf_sub_stats = hf_stats_all[workflow_idx]['submissions'][submission_idx]

        # Every third hpcflow task, since there are two additional hpcflow tasks for
        # each matflow task:
        hf_task_stats = hf_sub_stats['command_group_submissions'][1::3][task_idx]['tasks']
        task.resource_usage = hf_task_stats

        for elem_idx in range(num_elems):

            task_elem_path = self.get_element_path(task.task_idx, elem_idx)

            if task.schema.is_func:
                func_outputs_path = self._get_element_temp_output_path(task_idx, elem_idx)
                with func_outputs_path.open('rb') as handle:
                    func_outputs = pickle.load(handle)
                outputs[elem_idx].update(**func_outputs)
                func_outputs_path.unlink()

            # For each output to be parsed, invoke the function:
            for out_map in task.schema.output_map:

                # Filter only those file paths required for this output:
                file_paths = []
                for i in out_map['files']:
                    out_file_path = task_elem_path.joinpath(i['name'])
                    file_paths.append(out_file_path)

                    # Save generated file as string in workflow:
                    if i['save']:
                        with out_file_path.open('r') as handle:
                            task.files[elem_idx].update({i['name']: handle.read()})

                func = out_map_lookup[out_map['output']]
                output = func(*file_paths, **task.output_map_options)
                outputs[elem_idx].update({out_map['output']: output})

            # Save output files specified explicitly as outputs:
            for output_name in task.schema.outputs:
                if output_name.startswith('__file__'):
                    file_name = Config.get('output_file_maps')[schema_id].get(output_name)
                    if not file_name:
                        msg = 'Output file map missing for output name: "{}"'
                        raise ValueError(msg.format(output_name))
                    out_file_path = task_elem_path.joinpath(file_name)

                    # Save file in workflow:
                    with out_file_path.open('r') as handle:
                        outputs[elem_idx].update({output_name: handle.read()})

        task.outputs = outputs
        self._append_history(WorkflowAction.process_task)
