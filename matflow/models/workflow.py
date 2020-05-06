"""matflow.models.workflow.py

Module containing the Workflow class and some functions used to decorate Workflow methods.

"""

import copy
import functools
import secrets
from pathlib import Path
from pprint import pprint
from subprocess import run, PIPE
from warnings import warn

import hickle
import numpy as np
import yaml
from hpcflow.api import get_stats as hpcflow_get_stats

from matflow import (
    TASK_INPUT_MAP,
    TASK_OUTPUT_MAP,
    COMMAND_LINE_ARG_MAP,
    TASK_OUTPUT_FILES_MAP,
    __version__,
)
from matflow.errors import (
    IncompatibleTaskNesting,
    MissingMergePriority,
    WorkflowPersistenceError
)
from matflow.jsonable import to_jsonable
from matflow.utils import parse_times, zeropad
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


def increments_version(func):
    """Workflow method decorator to increment the workflow version when a method is
    successfully invoked."""
    @functools.wraps(func)
    def func_wrap(self, *args, **kwargs):
        ret = func(self, *args, **kwargs)
        self._version += 1
        return ret
    return func_wrap


def save_workflow(func):
    'Workflow method decorator to save the workflow after completion of the method.'
    @functools.wraps(func)
    def func_wrap(self, *args, **kwargs):
        ret = func(self, *args, **kwargs)
        self.save()
        return ret
    return func_wrap


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
        '_matflow_version',
        '_version',
    ]

    def __init__(self, name, tasks, stage_directory=None, extend=None,
                 check_integrity=True, __is_from_file=False, ):

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

        self._matflow_version = None
        self._version = None
        if not self.is_from_file:
            self._matflow_version = __version__
            self._version = 0

    def set_ids(self):
        if self._id:
            raise ValueError(f'IDs are already set for workflow. ID is: "{self.id}"; '
                             f'human ID is "{self.human_id}".')
        else:
            self._human_id = self.name_safe + '_' + parse_times('%Y-%m-%d-%H%M%S')[0]
            self._id = secrets.token_hex(15)

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

    @property
    def matflow_version(self):
        return self._matflow_version

    @property
    def version(self):
        return self._version

    @increments_version
    def write_directories(self):
        'Generate task and element directories.'

        if self.path.exists():
            raise ValueError('Directories for this workflow already exist.')

        self.path.mkdir(exist_ok=False)

        for elems_idx, task in zip(self.elements_idx, self.tasks):

            # Generate task directory:
            task_path = task.get_task_path(self.path)
            task_path.mkdir()

            num_elems = elems_idx['num_elements']
            # Generate element directories:
            for i in range(num_elems):
                task_elem_path = task_path.joinpath(str(zeropad(i, num_elems - 1)))
                task_elem_path.mkdir()

    @requires_path_exists
    def write_hpcflow_workflow(self):
        'Generate an hpcflow workflow file to execute this workflow.'

        command_groups = []
        variables = {}
        for elems_idx, task in zip(self.elements_idx, self.tasks):

            task_path_rel = str(task.get_task_path(self.path).name)

            # `input_vars` are those inputs that appear directly in the commands:
            fmt_commands, input_vars = task.schema.command_group.get_formatted_commands(
                task.local_inputs['inputs'].keys())

            cmd_line_inputs = {}
            for local_in_name, local_in in task.local_inputs['inputs'].items():
                if local_in_name in input_vars:
                    # TODO: We currently only consider input_vars for local inputs.

                    # Expand values for intra-task nesting:
                    values = [local_in['vals'][i] for i in local_in['vals_idx']]

                    # Format values:
                    fmt_func_scope = COMMAND_LINE_ARG_MAP.get(
                        (task.schema.name, task.schema.method, task.schema.implementation)
                    )
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

            num_elems = elems_idx['num_elements']

            task_path = task.get_task_path(self.path)

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
                for i in range(num_elems):

                    task_elem_path = task_path.joinpath(str(zeropad(i, num_elems - 1)))
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

            # (SGE specific)
            process_so = {'l': 'short'}
            sources = task.software_instance.get('sources', [])
            command_groups.extend([
                {
                    'directory': '.',
                    'nesting': 'hold',
                    'commands': [
                        'matflow prepare-task --task-idx={}'.format(task.task_idx)
                    ],
                    'stats': False,
                    'scheduler_options': process_so,
                },
                {
                    'directory': '<<{}_dirs>>'.format(task_path_rel),
                    'nesting': 'hold',
                    'commands': fmt_commands,
                    'sources': sources,
                    'stats': task.stats,
                    'scheduler_options': scheduler_opts,
                },
                {
                    'directory': '.',
                    'nesting': 'hold',
                    'commands': [
                        'matflow process-task --task-idx={}'.format(task.task_idx)
                    ],
                    'stats': False,
                    'scheduler_options': process_so,
                },
            ])

            # Add variable for the task directories:
            variables.update({
                '{}_dirs'.format(task_path_rel): {
                    'file_regex': {
                        'pattern': '({}/[0-9]+$)'.format(task_path_rel),
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

        with self.path.joinpath('1.hf.yml').open('w') as handle:
            yaml.safe_dump(hf_data, handle)

    def get_extended_workflows(self):
        if self.extend_paths:
            return [Workflow.load(i, full_path=True) for i in self.extend_paths]
        else:
            return None

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

        err_msg = None
        try:
            obj_json = to_jsonable(self, exclude=['_is_from_file'])
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
        except:
            msg = f'Could not load workflow file with `hickle`: "{path}".'
            raise WorkflowPersistenceError(msg)

        extend = None
        if obj_json['_extend_paths']:
            extend = {
                'paths': obj_json['_extend_paths'],
                'nest_idx': obj_json['_extend_nest_idx']
            }

        tasks = [{k.lstrip('_'): v for k, v in i.items()} for i in obj_json['_tasks']]
        obj = {
            'name': obj_json['_name'],
            'tasks': tasks,
            'stage_directory': obj_json['_stage_directory'],
            'extend': extend,
        }

        workflow = cls(
            **obj,
            _Workflow__is_from_file=True,
            check_integrity=check_integrity,
        )

        workflow.profile_str = obj_json['_profile_str']
        workflow._human_id = obj_json['_human_id']
        workflow._id = obj_json['_id']
        workflow._matflow_version = obj_json['_matflow_version']
        workflow._version = obj_json['_version']

        return workflow

    @save_workflow
    @increments_version
    @requires_path_exists
    def prepare_task(self, task_idx):
        'Prepare inputs and run input maps.'

        task = self.tasks[task_idx]
        elems_idx = self.elements_idx[task_idx]
        num_elems = elems_idx['num_elements']
        inputs = [{} for _ in range(num_elems)]
        files = [{} for _ in range(num_elems)]

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

        schema_id = (task.name, task.method, task.software)
        in_map_lookup = TASK_INPUT_MAP.get(schema_id)
        task_path = task.get_task_path(self.path)
        for elem_idx, elem_inputs in zip(range(num_elems), task.inputs):

            task_elem_path = task_path.joinpath(str(zeropad(elem_idx, num_elems - 1)))

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

    @save_workflow
    @increments_version
    @requires_path_exists
    def process_task(self, task_idx):
        'Process outputs from an executed task: run output map and save outputs.'

        task = self.tasks[task_idx]
        elems_idx = self.elements_idx[task_idx]
        num_elems = elems_idx['num_elements']
        outputs = [{} for _ in range(num_elems)]

        schema_id = (task.name, task.method, task.software)
        out_map_lookup = TASK_OUTPUT_MAP.get(schema_id)
        task_path = task.get_task_path(self.path)

        # Save hpcflow task stats
        hf_stats_all = hpcflow_get_stats(self.path, jsonable=True, datetime_dicts=True)

        print('matflow.models.workflow. process_task: hf_stats_all: ')
        pprint(hf_stats_all)

        workflow_idx = 0
        submission_idx = 0
        hf_sub_stats = hf_stats_all[workflow_idx]['submissions'][submission_idx]

        # Every third hpcflow task, since there are two additional hpcflow tasks for
        # each matflow task:
        hf_task_stats = hf_sub_stats['command_group_submissions'][1::3][task_idx]['tasks']
        task.resource_usage = hf_task_stats

        for elem_idx in range(num_elems):

            task_elem_path = task_path.joinpath(str(zeropad(elem_idx, num_elems - 1)))

            # For each output to be parsed, invoke the function:
            for out_map in task.schema.output_map:

                # Filter only those file paths required for this output:
                file_paths = []
                for i in out_map['files']:
                    out_file_path = task_elem_path.joinpath(i)
                    file_paths.append(out_file_path)

                    # Save generated file as string in workflow:
                    with out_file_path.open('r') as handle:
                        task.files[elem_idx].update({i: handle.read()})

                func = out_map_lookup[out_map['output']]
                output = func(*file_paths)
                outputs[elem_idx][out_map['output']] = output

            # Save output files specified explicitly as outputs:
            for output_name in task.schema.outputs:
                if output_name.startswith('__file__'):
                    file_name = TASK_OUTPUT_FILES_MAP[schema_id].get(output_name)
                    if not file_name:
                        msg = 'Output file map missing for output name: "{}"'
                        raise ValueError(msg.format(output_name))
                    out_file_path = task_elem_path.joinpath(file_name)

                    # Save file in workflow:
                    with out_file_path.open('r') as handle:
                        outputs[elem_idx][output_name] = handle.read()

        task.outputs = outputs
