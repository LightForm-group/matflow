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

import h5py
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
    UnexpectedSourceMapReturnError,
)
from matflow.hicklable import to_hicklable
from matflow.utils import parse_times, zeropad, datetime_to_dict
from matflow.models.command import DEFAULT_FORMATTERS
from matflow.models.construction import init_tasks
from matflow.models.software import SoftwareInstance
from matflow.models.task import TaskStatus


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
        '_loaded_path',
        '_profile',
        '_is_from_file',
        '_name',
        '_extends',
        '_stage_directory',
        '_tasks',
        '_elements_idx',
        '_history',
    ]

    def __init__(self, name, tasks, stage_directory=None, extends=None,
                 check_integrity=True, profile=None, __is_from_file=False):

        self._id = None             # Assigned once by set_ids()
        self._human_id = None       # Assigned once by set_ids()
        self._loaded_path = None    # Assigned on save or on load.

        self._is_from_file = __is_from_file
        self._name = name
        self._extends = [str(Path(i).resolve()) for i in (extends or [])]
        self._stage_directory = str(Path(stage_directory or '').resolve())
        self._profile = profile

        tasks, elements_idx = init_tasks(self, tasks, self.is_from_file, check_integrity)
        self._tasks = tasks
        self._elements_idx = elements_idx

        if not self.is_from_file:
            self._history = []
            self._append_history(WorkflowAction.generate)

    def __repr__(self):
        out = (
            f'{self.__class__.__name__}('
            f'human_id={self.human_id!r}, '
            f'version={self.version}'
            f')'
        )
        return out

    def __str__(self):
        tasks = '\n'.join([
            (
                f'  {i.name}\n'
                f'    Method: {i.method}\n'
                f'    Software: {i.software}\n'
            )
            for i in self.tasks
        ])
        out = (
            f'{"ID:":10}{self.id!s}\n'
            f'{"Name:":10}{self.name!s}\n'
            f'{"Human ID:":10}{self.human_id!s}\n'
            f'Tasks:\n\n{tasks}'
        )
        return out

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
    def loaded_path(self):
        return Path(self._loaded_path)

    @loaded_path.setter
    def loaded_path(self, loaded_path):
        if not loaded_path.is_file():
            raise TypeError('`loaded_path` is not a file.')
        self._loaded_path = loaded_path

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
    def name_friendly(self):
        'Capitalise and remove underscores'
        name = '{}{}'.format(self.name[0].upper(), self.name[1:]).replace('_', ' ')
        return name

    @property
    def profile_file(self):
        return self._profile['file']

    @property
    def profile(self):
        return self._profile['parsed']

    @property
    def tasks(self):
        return self._tasks

    @property
    def elements_idx(self):
        return self._elements_idx

    @property
    def extends(self):
        return [Path(i) for i in self._extends]

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

    @property
    def default_file_path(self):
        return self.path.joinpath('workflow.hdf5')

    @property
    def HDF5_path(self):
        return '/workflow_obj/data_0'

    @functools.lru_cache
    def get_element_data(self, idx):
        with h5py.File(self.loaded_path, 'r') as handle:
            path = f'/element_data/data_0'
            num_dat = len(handle[path])
            if idx > (num_dat - 1):
                raise ValueError(f'Element data has {num_dat} member(s), but idx={idx} '
                                 f'requested.')
            dat_path = path + f'/{idx}'
            return hickle.load(handle, path=dat_path)

    def get_task_idx_padded(self, task_idx, ret_zero_based=True):
        'Get a task index, zero-padded according to the number of tasks.'
        if ret_zero_based:
            return zeropad(task_idx, len(self) - 1)
        else:
            return zeropad(task_idx + 1, len(self))

    @requires_path_exists
    def get_task_path(self, task_idx):
        'Get the path to a task directory.'
        if task_idx > (len(self) - 1):
            msg = f'Workflow has only {len(self)} tasks.'
            raise ValueError(msg)
        task = self.tasks[task_idx]
        task_idx_fmt = self.get_task_idx_padded(task_idx, ret_zero_based=False)
        task_path = self.path.joinpath(f'task_{task_idx_fmt}_{task.name}')
        return task_path

    @requires_path_exists
    def get_task_sources_path(self, task_idx):
        task_path = self.get_task_path(task_idx)
        return task_path.joinpath('sources')

    @requires_path_exists
    def get_element_path(self, task_idx, element_idx):
        'Get the path to an element directory.'
        num_elements = self.elements_idx[task_idx]['num_elements']
        if element_idx > (num_elements - 1):
            msg = f'Task at index {task_idx} has only {num_elements} elements.'
            raise ValueError(msg)

        element_path = self.get_task_path(task_idx)
        if num_elements > 1:
            element_idx_fmt = f'element_{zeropad(element_idx + 1, num_elements)}'
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

            if task.software_instance.requires_sources:
                self.get_task_sources_path(task.task_idx).mkdir()

            num_elems = elems_idx['num_elements']
            # Generate element directories:
            for i in range(num_elems):
                self.get_element_path(task.task_idx, i).mkdir(exist_ok=True)

    def get_hpcflow_job_name(self, task, job_type, is_stats=False):
        """Get the scheduler job name for a given task index and job type.

        Parameters
        ----------
        task : Task
        job_type : str 
            One of "prepare-task", "process-task", "run", "prepare-sources"
        is_stats : bool, optional

        Returns
        -------
        job_name : str
            The job name to be used in the hpcflow workflow.

        """
        ALLOWED = ['prepare-task', 'process-task', 'run', 'prepare-sources']
        if job_type not in ALLOWED:
            raise ValueError(f'Invalid `job_type`. Allowed values are: {ALLOWED}.')

        task_idx_fmt = self.get_task_idx_padded(task.task_idx, ret_zero_based=False)

        base = 't' if not is_stats else 's'

        if job_type == 'run':
            out = f'{base}{task_idx_fmt}'

        elif job_type == 'prepare-task':
            if task.task_idx == 0:
                out = f'{base}{task_idx_fmt}_aux'
            else:
                prev_task = self.tasks[task.task_idx - 1]
                prev_task_idx_fmt = self.get_task_idx_padded(
                    prev_task.task_idx,
                    ret_zero_based=False,
                )
                out = f't{prev_task_idx_fmt}+{task_idx_fmt}_aux'

        elif job_type == 'process-task':
            if task.task_idx == (len(self) - 1):
                out = f'{base}{task_idx_fmt}_aux'
            else:
                next_task = self.tasks[task.task_idx + 1]
                next_task_idx_fmt = self.get_task_idx_padded(
                    next_task.task_idx,
                    ret_zero_based=False,
                )
                out = f'{base}{task_idx_fmt}+{next_task_idx_fmt}_aux'

        elif job_type == 'prepare-sources':
            out = f'{base}{task_idx_fmt}_src'

        return out

    @requires_path_exists
    def get_hpcflow_workflow(self):
        'Generate an hpcflow workflow to execute this workflow.'

        command_groups = []
        variables = {}
        for elems_idx, task in zip(self.elements_idx, self.tasks):

            src_prep_cmd_group = []
            executable = task.software_instance.executable
            scheduler_opts_process = Config.get('prepare_process_scheduler_options')

            if task.software_instance.requires_sources:

                sources_dir = str(self.get_task_sources_path(task.task_idx))
                prep_env = task.software_instance.preparation['environment'] or ''
                prep_cmds = task.software_instance.preparation['commands'] or ''

                source_map = Config.get('sources_maps')[
                    (task.name, task.method, task.software)
                ]
                src_vars = source_map['sources']
                for src_var_name, src_name in src_vars.items():

                    prep_cmds = prep_cmds.replace(f'<<{src_var_name}>>', src_name)
                    executable = executable.replace(f'<<{src_var_name}>>', src_name)

                    prep_cmds = prep_cmds.replace('<<sources_dir>>', sources_dir)
                    executable = executable.replace('<<sources_dir>>', sources_dir)

                src_prep_cmd_group = [{
                    'directory': str(self.get_task_sources_path(task.task_idx)),
                    'nesting': 'hold',
                    'commands': [
                        'matflow prepare-sources --task-idx={}'.format(task.task_idx)
                    ] + prep_cmds.splitlines(),
                    'environment': prep_env.splitlines(),
                    'stats': False,
                    'scheduler_options': scheduler_opts_process,
                    'name': self.get_hpcflow_job_name(task, 'prepare-sources'),
                }]

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

                fmt_commands = [i.replace('<<executable>>', executable)
                                for i in fmt_commands]

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
            for k, v in task.run_options.items():
                if k != 'num_cores':
                    if k == 'pe':
                        v = v + ' ' + str(task.run_options['num_cores'])
                    scheduler_opts.update({k: v})

            task_path_rel = str(self.get_task_path(task.task_idx).name)

            if task.task_idx == 0:
                command_groups.append({
                    'directory': '.',
                    'nesting': 'hold',
                    'commands': task.prepare_task_commands,
                    'stats': False,
                    'scheduler_options': scheduler_opts_process,
                    'name': self.get_hpcflow_job_name(task, 'prepare-task'),
                })

            if task.software_instance.requires_sources:
                command_groups += src_prep_cmd_group

            command_groups.append({
                'directory': '<<{}_dirs>>'.format(task_path_rel),
                'nesting': 'hold',
                'commands': fmt_commands,
                'environment': task.software_instance.environment_lines,
                'scheduler_options': scheduler_opts,
                'name': self.get_hpcflow_job_name(task, 'run'),
                'stats': task.stats,
                'stats_name': self.get_hpcflow_job_name(task, 'run', is_stats=True),
            })

            if task.task_idx < (len(self) - 1):
                next_task = self.tasks[task.task_idx + 1]
                command_groups.append({
                    'directory': '.',
                    'nesting': 'hold',
                    'commands': (
                        task.process_task_commands + next_task.prepare_task_commands
                    ),
                    'stats': False,
                    'scheduler_options': scheduler_opts_process,
                    'name': self.get_hpcflow_job_name(task, 'process-task'),
                })
            else:
                command_groups.append({
                    'directory': '.',
                    'nesting': 'hold',
                    'commands': task.process_task_commands,
                    'stats': False,
                    'scheduler_options': scheduler_opts_process,
                    'name': self.get_hpcflow_job_name(task, 'process-task'),
                })

            # Add variable for the task directories:
            elem_dir_regex = '/element_[0-9]+$' if elems_idx['num_elements'] > 1 else ''
            variables.update({
                '{}_dirs'.format(task_path_rel): {
                    'file_regex': {
                        'pattern': f'({task_path_rel}{elem_dir_regex})$',
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
        out['tasks'] = [i.as_dict() for i in out['tasks']]

        del out['is_from_file']
        del out['loaded_path']

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
            raise TypeError(f'The following path is not a directory: {directory}.')

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

    @classmethod
    def get_workflow_files(cls, directory):

        directory = Path(directory)
        if not directory.is_dir():
            raise TypeError(f'The following path is not a directory: {directory}.')

        existing_files = {}
        for i in directory.glob('*'):
            if i.is_file():
                try:
                    with h5py.File(i, 'r') as handle:
                        id_ = handle.attrs['workflow_id']
                        version = handle.attrs['workflow_version']
                except Exception as err:
                    continue
                existing_files.update({i: {'id': id_, 'version': version}})

        return existing_files

    def write_HDF5_file(self, path=None):
        """Save the initial workflow to an HDF5 file.

        Parameters
        ----------
        path : str or Path, optional
            If specified, must be the full path (including file name) where the workflow
            file should be saved. By default, `None`, in which case the
            `default_file_path` attribute will be used as the full path.

        """

        path = Path(path or self.default_file_path)
        if path.exists():
            msg = f'Workflow cannot be saved to a path that already exists: "{path}".'
            raise WorkflowPersistenceError(msg)

        workflow_as_dict = self.as_dict()
        obj_json = to_hicklable(workflow_as_dict)

        with h5py.File(path, 'w-') as handle:

            handle.attrs['matflow_version'] = __version__
            handle.attrs['workflow_id'] = self.id
            handle.attrs['workflow_version'] = 0

            workflow_group = handle.create_group('workflow_obj')
            workflow_group.attrs['type'] = [b'hickle']
            hickle.dump(obj_json, handle, path=workflow_group.name)

            data_group = handle.create_group('element_data')
            data_group.attrs['type'] = [b'hickle']
            hickle.dump({}, handle, path=data_group.name)

        self.loaded_path = path

    @classmethod
    def load_HDF5_file(cls, path=None, full_path=False, check_integrity=True):
        """Load workflow from an HDF5 file.

        Parameters
        ----------
        path : str or Path
            Either the directory in which to search for a suitable workflow file (if
            `full_path=False`), or the full path to a workflow file (if `full_path=True`).
            If multiple workflow files with distinct IDs exist in the loading directory,
            an exception is raised.
        full_path : bool, optional
            Determines whether `path` is a full workflow file path or a directory path.
            By default, False.
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
            existing = cls.get_workflow_files(path)
            if not existing:
                raise ValueError('No workflow files found.')
            all_IDs = set([v['id'] for v in existing.values()])
            if len(all_IDs) > 1:
                msg = (f'Saved workflows with multiple distinct IDs exist in the loading'
                       f' directory "{path}". Specify `path` as the full path to the '
                       f'workflow file, and set `full_path=True`.')
                raise WorkflowPersistenceError(msg)
            else:
                # Get full path of workflow file with the largest version number:
                path = sorted(existing.items(), key=lambda i: i[1]['version'])[-1][0]
        try:
            with h5py.File(path, 'r') as handle:
                obj_json = hickle.load(handle, path='/workflow_obj')
        except Exception as err:
            msg = f'Could not load workflow object with `hickle`: "{path}": {err}.'
            raise WorkflowPersistenceError(msg)

        for i in obj_json['tasks']:
            i['status'] = TaskStatus(i['status'][1])
            soft_inst_dict = i['software_instance']
            machine = soft_inst_dict.pop('machine')
            soft_inst = SoftwareInstance(**soft_inst_dict)
            soft_inst.machine = machine
            i['software_instance'] = soft_inst

        obj = {
            'name': obj_json['name'],
            'tasks': obj_json['tasks'],
            'stage_directory': obj_json['stage_directory'],
            'profile': obj_json['profile'],
            'extends': obj_json['extends'],
        }

        workflow = cls(
            **obj,
            _Workflow__is_from_file=True,
            check_integrity=check_integrity,
        )

        workflow._human_id = obj_json['human_id']
        workflow._id = obj_json['id']

        for i in obj_json['history']:
            i['action'] = WorkflowAction(i['action'][1])
            i['timestamp'] = datetime(**i['timestamp'])
        workflow._history = obj_json['history']
        workflow.loaded_path = path

        return workflow

    @requires_path_exists
    def prepare_sources(self, task_idx):
        'Prepare source files for the task preparation commands.'

        # Note: in future, we might want to parametrise the source function, which is
        # why we delay its invocation until task run time.

        task = self.tasks[task_idx]

        if not task.software_instance.requires_sources:
            raise RuntimeError('The task has no sources to prepare.')

        source_map = Config.get('sources_maps')[(task.name, task.method, task.software)]
        source_func = source_map['func']
        source_files = source_func()

        print(f'source_files:\n{source_files}')

        expected_src_vars = set(source_map['sources'].keys())
        returned_src_vars = set(source_files.keys())
        bad_keys = returned_src_vars - expected_src_vars
        if bad_keys:
            bad_keys_fmt = ', '.join([f'"{i}"' for i in bad_keys])
            msg = (f'The following source variable names were returned by the sources '
                   f'mapper function "{source_func}", but were not expected: '
                   f'{bad_keys_fmt}.')
            raise UnexpectedSourceMapReturnError(msg)

        miss_keys = expected_src_vars - returned_src_vars
        if miss_keys:
            miss_keys_fmt = ', '.join([f'"{i}"' for i in miss_keys])
            msg = (f'The following source variable names were not returned by the sources'
                   f' mapper function "{source_func}": {miss_keys_fmt}.')
            raise UnexpectedSourceMapReturnError(msg)

        for src_var, src_name in source_map['sources'].items():
            file_str = source_files[src_var]['content']
            file_name = source_files[src_var]['filename']
            file_path = self.get_task_sources_path(task_idx).joinpath(file_name)
            with file_path.open('w') as handle:
                handle.write(file_str)

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

        try:
            # Get software versions:
            software_versions_func = Config.get('software_versions').get(task.software)
            if software_versions_func:
                if task.schema.is_func:
                    software_versions = software_versions_func()
                else:
                    executable = task.software_instance.executable
                    software_versions = software_versions_func(executable)
            else:
                software_versions = task.software_instance.version_info
        except Exception as err:
            software_versions = None
            warn(f'Failed to parse software versions: {err}')

        task.status = TaskStatus.running
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
        job_name = self.get_hpcflow_job_name(task, 'run')
        for i in hf_sub_stats['command_group_submissions']:
            if i['name'] == job_name:
                task.resource_usage = i['tasks']
                break

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
        task.status = TaskStatus.complete
        self._append_history(WorkflowAction.process_task)
