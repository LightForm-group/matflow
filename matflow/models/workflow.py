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

from matflow import (CONFIG, SOFTWARE, TASK_SCHEMAS, TASK_INPUT_MAP,
                     TASK_OUTPUT_MAP, TASK_FUNC_MAP, COMMAND_LINE_ARG_MAP,
                     TASK_OUTPUT_FILES_MAP, __version__)
from matflow.command_formatters import DEFAULT_FORMATTERS
from matflow.models import Task
from matflow.models.task import (get_schema_dict, combine_base_sequence, TaskSchema,
                                 get_local_inputs)
from matflow.jsonable import to_jsonable
from matflow.utils import parse_times, zeropad
from matflow.errors import (
    IncompatibleWorkflow,
    IncompatibleTaskNesting,
    MissingMergePriority,
    MissingSoftware,
    WorkflowPersistenceError
)


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
                 __is_from_file=False):

        self._id = None             # Assigned once by set_ids()
        self._human_id = None       # Assigned once by set_ids()
        self._profile_str = None    # Assigned once in `profile_str` setter

        self._is_from_file = __is_from_file
        self._name = name
        self._extend_paths = [str(Path(i).resolve())
                              for i in extend['paths']] if extend else None
        self._extend_nest_idx = extend['nest_idx'] if extend else None
        self._stage_directory = str(Path(stage_directory or '').resolve())

        tasks, elements_idx = self._validate_tasks(tasks)
        self._tasks = tasks
        self._elements_idx = elements_idx

        self._matflow_version = None
        self._version = None
        if not self.is_from_file:
            self._matflow_version = __version__
            self._version = 0

    def _validate_tasks(self, tasks):

        # TODO: validate sequences dicts somewhere.

        task_info_lst = []
        software_instances = []
        for task in tasks:

            software_instance = task.get('software_instance')
            if task.get('software'):
                if not software_instance:
                    software_instance = self._get_software_instance(
                        task['software'],
                        task['run_options'].get('num_cores', 1),
                    )
                else:
                    software_instance['num_cores'] = [
                        int(i) for i in software_instance['num_cores']]

            software_instances.append(software_instance)
            schema_dict = get_schema_dict(task['name'], task['method'], software_instance)
            schema = TaskSchema(**schema_dict)

            local_inputs = task.get('local_inputs')
            if local_inputs is None:
                local_inputs = get_local_inputs(
                    base=task.get('base'),
                    num_repeats=task.get('num_repeats'),
                    sequences=task.get('sequences'),
                )

            task_info_lst.append({
                'name': task['name'],
                'inputs': schema.inputs,
                'outputs': schema.outputs,
                'length': local_inputs['length'],
                'nest': task.get('nest'),
                'merge_priority': task.get('merge_priority'),
                'schema': schema,
                'local_inputs': local_inputs,
            })

        task_srt_idx, task_info_lst, elements_idx = check_task_compatibility(
            task_info_lst)

        validated_tasks = []
        for idx, i in enumerate(task_srt_idx):

            # Reorder and instantiate task
            task_i = tasks[i]

            task_i.pop('base', None)
            task_i.pop('sequences', None)
            task_i.pop('software', None)

            task_i['nest'] = task_info_lst[idx]['nest']
            task_i['task_idx'] = task_info_lst[idx]['task_idx']
            task_i['merge_priority'] = task_info_lst[idx]['merge_priority']
            task_i['software_instance'] = software_instances[idx]
            task_i['schema'] = task_info_lst[idx]['schema']
            task_i['local_inputs'] = task_info_lst[idx]['local_inputs']

            task_i_obj = Task(**task_i)
            validated_tasks.append(task_i_obj)

        return validated_tasks, elements_idx

    def _get_software_instance(self, software_name, num_cores):
        """Find a software instance in the software.yml file that matches the software
        requirements of a given task."""

        for soft_inst in SOFTWARE:

            if soft_inst['name'] != software_name:
                continue

            core_range = soft_inst['num_cores']
            all_num_cores = list(range(*core_range)) + [core_range[1]]
            if num_cores in all_num_cores:
                return soft_inst

        raise MissingSoftware(f'Could not find suitable software "{software_name}", with'
                              f' `num_cores={num_cores}`.')

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

            # `input_vars` are those inputs that appear in the commands:
            fmt_commands, input_vars = task.schema.command_group.get_formatted_commands(
                task.local_inputs['inputs'].keys())

            cmd_line_inputs = {}
            for local_in_name, local_in in task.local_inputs['inputs'].items():
                if local_in_name in input_vars:

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

            sources = task.software_instance.get('sources', [])
            command_groups.extend([
                {
                    'directory': '.',
                    'nesting': 'hold',
                    'commands': [
                        'matflow prepare-task --task-idx={}'.format(task.task_idx)
                    ],
                    'stats': False,
                },
                {
                    'directory': '<<{}_dirs>>'.format(task_path_rel),
                    'nesting': 'hold',
                    'commands': fmt_commands,
                    'sources': sources,
                    'stats': task.stats,
                },
                {
                    'directory': '.',
                    'nesting': 'hold',
                    'commands': [
                        'matflow process-task --task-idx={}'.format(task.task_idx)
                    ],
                    'stats': False,
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

            if self.version in [i['version'] for i in to_delete.keys()]:
                warn('A saved workflow with the same ID and version already exists in '
                     'this directory. This will be removed.')

            # Mark older versions for deletion:
            for del_path_i in to_delete.keys():
                tagged_name = del_path_i.with_name(del_path_i.name + remove_tag)
                del_path_i.rename(tagged_name)

        if path.exists():
            msg = f'Workflow cannot be saved to a path that already exists: "{path}".'
            raise WorkflowPersistenceError(msg)

        err_msg = None
        try:
            obj_json = to_jsonable(self, exclude=['_is_from_file'])
            try:
                with path.open('w') as handle:
                    hickle.dump(obj_json, handle)
            except:
                err_msg = f'Failed to save workflow to path: "{path}".'
        except:
            err_msg = 'Failed to convert Workflow object to `hickle`-compatible dict.'

        if err_msg:
            if not keep_previous_versions:
                # Revert older versions back to original file names (don't delete):
                for del_path_i in to_delete.keys():
                    tagged_name = del_path_i.with_name(del_path_i.name + remove_tag)
                    tagged_name.rename(del_path_i)

                del to_delete
            raise WorkflowPersistenceError(err_msg)

        else:
            if not keep_previous_versions:
                # Delete older versions (same ID):
                for del_path_i in to_delete.keys():
                    tagged_name = del_path_i.with_name(del_path_i.name + remove_tag)
                    tagged_name.unlink(missing_ok=False)

    @classmethod
    def load(cls, path, full_path=False, version=None):
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
                'nest_idx': obj_json['extend_nest_idx']
            }

        obj = {
            'name': obj_json['name'],
            'tasks': obj_json['tasks'],
            'stage_directory': obj_json['_stage_directory'],
            'extend': extend,
        }

        workflow = cls(**obj, __is_from_file=True)

        workflow.profile_str = obj_json['profile_str']
        workflow._human_id = obj_json['human_id']
        workflow._id = obj_json['id']
        workflow._matflow_version = obj_json['matflow_version']
        workflow._version = obj_json['version']

        return workflow

    def proceed(self):
        'Start or continue the Workflow task execution.'

        print('Workflow.proceed')

        # If task is remote or scheduled, instead of executing the commands, generate
        # an hpcflow project, transfer to remote location and submit.
        # Task status will be 'waiting', the output_map will not yet have been run
        # So once we get to a 'waiting' task, we should try to run the output map
        # if successful, can continue with next task. Otherwise, end.

        allpaths = list(self.path.glob('*'))
        print(f'allpaths: {allpaths}')

        for task_idx, task in enumerate(self.tasks):

            print(f'task_idx: {task_idx}')

            if task.status == 'complete':
                continue

            elif task.status == 'waiting':
                # TODO: Run output map
                # TODO: set task.status to 'complete' and update workflow state
                continue

            elif task.status == 'pending':

                self.prepare_task_inputs(task_idx)
                task.initialise_outputs()
                task_path = task.get_task_path(self.path)
                task_path.mkdir()

                all_task_elem_path = []
                for element_idx in range(len(task)):

                    # print(f'i (element idx): {element_idx}')

                    input_props = task.inputs[element_idx]
                    # print('input_props:')
                    # pprint(input_props)

                    # Make a sub-dir for each element:
                    task_elem_path = task_path.joinpath(
                        str(zeropad(element_idx, len(task) - 1)))
                    task_elem_path.mkdir()
                    all_task_elem_path.append(task_elem_path)

                    # Copy any input files to task element directory:
                    input_file_keys = {}
                    for k, v in input_props.items():
                        if k.startswith('_file:'):
                            src_file_path = self.stage_directory.joinpath(v)
                            dst_file_path = task_elem_path.joinpath(src_file_path.name)
                            dst_file_path.write_bytes(src_file_path.read_bytes())
                            dst_path_rel = dst_file_path.relative_to(task_elem_path)
                            input_file_keys.update({k: str(dst_path_rel)})

                    # Rename input property without "_file:" prefix:
                    for k, new_v in input_file_keys.items():
                        new_k = k.split('_file:')[1]
                        input_props.update({new_k: new_v})
                        del input_props[k]

                    if task.schema.is_func:

                        args = copy.deepcopy(input_props)
                        output = TASK_FUNC_MAP[(task.name, task.method)](**args)

                        print('got output from func map: ')
                        pprint(output)

                        if output is not None:
                            # Assume for now that a function outputs only one
                            # "object/label":
                            task.outputs[element_idx][task.schema.outputs[0]] = output

                            print('updated task output:')
                            pprint(task.outputs)

                    elif task.schema.input_map:
                        # Run input map for each element

                        # For this task, get the input map function lookup:
                        in_map_lookup = TASK_INPUT_MAP[
                            (task.name, task.method, task.software)
                        ]

                        # For each input file to be written, invoke the function:
                        for in_map in task.schema.input_map:

                            func = in_map_lookup[in_map['file']]

                            for k_idx in range(len(in_map['inputs'])):
                                if in_map['inputs'][k_idx] in input_file_keys:
                                    new_k = in_map['inputs'][k_idx].split('_file:')[1]
                                    in_map['inputs'][k_idx] = new_k

                            # Filter only those inputs required for this file:
                            in_map_inputs = {
                                key: val for key, val in input_props.items()
                                if key in in_map['inputs']
                            }
                            file_path = task_elem_path.joinpath(in_map['file'])

                            # print('in_map_inputs:')
                            # pprint(in_map_inputs)

                            func(path=file_path, **in_map_inputs)

                            if task.inputs[element_idx].get('_files') is None:
                                task.inputs[element_idx].update({
                                    '_files': {},
                                })

                            # Add input file path as an input for the task
                            task.inputs[element_idx]['_files'].update({
                                in_map['file']: str(file_path)
                            })

                if not task.schema.is_func:

                    if task.is_scheduled:
                        # Make hpcflow project in task directory:
                        task.schema.command_group.prepare_scheduled_execution()
                        task.status = 'waiting'
                    else:
                        # Write execution "jobscript" in task directory:
                        run_commands = []
                        for element_idx in range(len(task)):
                            run_cmd = task.schema.command_group.prepare_direct_execution(
                                task.inputs[element_idx],
                                task.get_resource(self),
                                all_task_elem_path[element_idx],
                                task.software_instance.get('wsl_wrapper'),
                            )
                            run_commands.append(run_cmd)

                    if task.is_remote(self):
                        # Copy task directory to remote resource
                        # TODO:
                        task.status = 'waiting'
                        if task.pause:
                            print('Pausing task.')
                            break
                        if task.is_scheduled:
                            # Submit hpcflow project remotely over SSH
                            pass
                        else:
                            # Execute "jobscript" over SSH. # change perms to executable.
                            msg = 'Remote non-scheduled not yet supported.'
                            raise NotImplementedError(msg)
                    else:
                        if task.pause:
                            print('Pausing task.')
                            break
                        # Execute "jobscript" locally and wait.
                        for i in run_commands:
                            proc = run(i['run_cmd'], shell=True, stdout=PIPE, stderr=PIPE,
                                       cwd=str(task_path.joinpath(i['run_dir'])))

                            print('stdout', proc.stdout.decode())
                            print('stderr', proc.stderr.decode())

                    if task.status == 'waiting':
                        print('Task is waiting, pausing workflow progression.')
                        break

                    else:

                        for element_idx in range(len(task)):

                            if task.schema.output_map:
                                # For this task, get the output map function lookup:
                                out_map_lookup = TASK_OUTPUT_MAP[
                                    (task.name, task.method, task.software)]

                                # For each output to be parsed, invoke the function:
                                for out_map in task.schema.output_map:

                                    func = out_map_lookup[out_map['output']]

                                    # Add generated file paths to outputs:
                                    if task.outputs[element_idx].get('_files') is None:
                                        task.outputs[element_idx].update({
                                            '_files': {},
                                        })

                                    task_elem_path = all_task_elem_path[element_idx]

                                    # Filter only those file paths required for this output:
                                    file_paths = []
                                    for i in out_map['files']:
                                        out_file_path = task_elem_path.joinpath(i)
                                        file_paths.append(out_file_path)
                                        task.outputs[element_idx]['_files'].update({
                                            i: str(out_file_path),
                                        })

                                    output = func(*file_paths)
                                    # print(f'\nresolved output is: {output}\n')
                                    task.outputs[element_idx][out_map['output']] = output

                            # Add output files:
                            for outp in task.schema.outputs:
                                if outp.startswith('_file:'):
                                    # Add generated file paths to outputs:
                                    if task.outputs[element_idx].get('_files') is None:
                                        task.outputs[element_idx].update({
                                            '_files': {},
                                        })
                                    file_base = outp.split('_file:')[1]
                                    task.outputs[element_idx]['_files'].update({
                                        file_base: str(task_elem_path.joinpath(file_base))})

        # for i in self.tasks:
        #     print('task inputs')
        #     pprint(i.inputs)
        #     print('task outputs')
        #     pprint(i.outputs)

        self.save_state()

    def prepare_task_inputs(self, task_idx):
        """Prepare the inputs of a task that is about to be executed, by searching for
        relevant outputs from all previous tasks.

        Parameters
        ----------
        task_idx : int
            The index of the task that is about to be executed.

        """

        cur_task = self.tasks[task_idx]
        cur_task_ins = cur_task.schema.inputs

        # print(f'\nexpand_task_inputs.')
        # print(f'task_idx: {task_idx} cur_task_ins: {cur_task_ins}')

        if task_idx == 0:
            return

        collated_outputs = []
        for inp_idx, inp in enumerate(cur_task_ins):

            # Also search for outputs in tasks from extended workflow:
            ext_workflow_tasks = []
            if self.extend_paths:
                ext_workflow_tasks = [j for i in self.get_extended_workflows()
                                      for j in i.tasks]

            cur_workflow_tasks = [i for i in self.tasks if i.task_idx < task_idx]
            prev_task_list = ext_workflow_tasks + cur_workflow_tasks

            for prev_task in prev_task_list:

                prev_task_outs = prev_task.schema.outputs

                if inp in prev_task_outs:

                    if inp.startswith('_file:'):
                        file_name = inp.split('_file:')[1]
                        file_base = file_name.split('.')[0]
                        file_func_arg = f'{file_base}_path'
                        inp_name = file_func_arg
                        # print(f'file_name: {file_name}')
                        # print(f'file_base: {file_base}')
                        # print(f'file_func_arg: {file_func_arg}')
                        vals = [{file_func_arg: i['_files'][file_name]}
                                for i in prev_task.outputs]

                    else:
                        vals = [{inp: i[inp]} for i in prev_task.outputs]
                        inp_name = inp

                    # print(f'inp {inp} in prev_tasks_outs!')

                    new_in_nest_idx = prev_task.nest_idx
                    if self.extend_nest_idx:
                        if inp in self.extend_nest_idx:
                            new_in_nest_idx = self.extend_nest_idx[inp]
                    new_inputs = {
                        'name': inp_name,
                        'nest_idx': new_in_nest_idx,
                        'vals': vals,
                    }

                    # print(f'new_inputs: \n{new_inputs}')
                    collated_outputs.append(new_inputs)

        # Coerce original task inputs into form suitable for combining
        # with outputs of previous tasks:
        task_inputs = [{
            'name': '_inputs',
            'nest_idx': cur_task.nest_idx,
            'vals': cur_task.inputs
        }]

        # print(f'task_inputs: \n{task_inputs}')

        # print(f'will now combine base sequence with collated_outputs: '
        #       f'\n{collated_outputs}\n and task_inputs: \n{task_inputs}')

        # Now combine collated outputs with next task's inputs according to
        # respective `nest_idx`s to form expanded inputs:
        expanded_ins = combine_base_sequence(collated_outputs + task_inputs)

        # Reassign task inputs:
        cur_task.inputs = expanded_ins

    @requires_path_exists
    @increments_version
    def prepare_task(self, task_idx):
        'Prepare inputs and run input maps.'

        task = self.tasks[task_idx]
        elems_idx = self.elements_idx[task_idx]
        num_elems = elems_idx['num_elements']
        inputs = [{} for _ in range(num_elems)]
        files = [{} for _ in range(num_elems)]

        for input_name, inputs_idx in elems_idx['inputs'].items():
            task_idx = inputs_idx.get('task_idx')
            if task_idx is not None:
                # Input values should be copied from a previous task's `outputs`
                prev_task = self.tasks[task_idx]
                prev_outs = prev_task.outputs

                if not prev_outs:
                    msg = ('Task "{}" does not have the outputs required to parametrise '
                           'the current task: "{}".')
                    raise ValueError(msg.format(prev_task.name, task.name))

                values_all = [prev_outs[i][input_name] for i in inputs_idx['output_idx']]

            else:
                # Input values should be copied from this task's `local_inputs`

                # Expand values for intra-task nesting:
                local_in = task.local_inputs['inputs'][input_name]
                values = [local_in['vals'][i] for i in local_in['vals_idx']]

                # Expand values for inter-task nesting:
                values_all = [values[i] for i in inputs_idx['input_idx']]

            for element, val in zip(inputs, values_all):
                element.update({input_name: val})

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

        self.save()

    @requires_path_exists
    @increments_version
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

        self.save()


def check_task_compatibility(task_info_lst):
    'Check workflow has no incompatible tasks.'

    """
    TODO: 
        * enforce restriction: any given output from one task may only be used as
            input in _one_ other task.
        * When considering nesting, specify `nest: True | False` on the outputting
            task (not the inputting task). Must do this since
        * when extending a workflow, need to also specify which outputs to extract and
            whether `nest: True | False`.
        * implement new key: `merge_priority: INT 0 -> len(outputting tasks)-1`:
            * must be specified on all outputting tasks if any of them is `nest: False`
            * if not specified (and all are `nest: True`), default will be set as
                randomly range(len(outputting tasks)-1).

    """

    dependency_idx = get_dependency_idx(task_info_lst)
    check_missing_inputs(task_info_lst, dependency_idx)

    # Find the index at which each task must be positioned to satisfy input
    # dependencies, and reorder tasks (and `dependency_idx`!):
    min_idx = [max(i or [0]) + 1 for i in dependency_idx]
    task_srt_idx = np.argsort(min_idx)
    task_info_lst = [task_info_lst[i] for i in task_srt_idx]
    dependency_idx = [[np.argsort(task_srt_idx)[j] for j in dependency_idx[i]]
                      for i in task_srt_idx]

    # Add sorted task idx:
    for idx, i in enumerate(task_info_lst):
        i['task_idx'] = idx

    # Note: when considering upstream tasks for a given downstream task, need to nest
    # according to the upstream tasks' `num_elements`, not their `length`.
    elements_idx = []
    for idx, downstream_tsk in enumerate(task_info_lst):

        # Do any further downstream tasks depend on this task?
        depended_on = False
        for deps_idx in dependency_idx[(idx + 1):]:
            if idx in deps_idx:
                depended_on = True
                break

        if not depended_on and downstream_tsk.get('nest') is not None:
            msg = '`nest` value is specified but not required for task "{}".'
            warn(msg.format(downstream_tsk['name']))

        # Add `nest: True` by default to this task if nesting is not specified, and if at
        # least one further downstream task depends on this task:
        if downstream_tsk.get('nest', None) is None:
            if depended_on:
                downstream_tsk['nest'] = True

        # Add default `merge_priority` of `None`:
        if 'merge_priority' not in downstream_tsk:
            downstream_tsk['merge_priority'] = None

        upstream_tasks = [task_info_lst[i] for i in dependency_idx[idx]]
        num_elements = get_task_num_elements(downstream_tsk, upstream_tasks)
        downstream_tsk['num_elements'] = num_elements

        task_elems_idx = get_task_elements_idx(downstream_tsk, upstream_tasks)

        params_idx = get_input_elements_idx(task_elems_idx, downstream_tsk, task_info_lst)
        elements_idx.append({
            'num_elements': num_elements,
            'inputs': params_idx,
        })

    return list(task_srt_idx), task_info_lst, elements_idx


def check_missing_inputs(task_info_lst, dependency_list):

    for deps_idx, task_info in zip(dependency_list, task_info_lst):

        defined_inputs = list(task_info['local_inputs']['inputs'].keys())
        task_info['schema'].check_surplus_inputs(defined_inputs)

        if deps_idx:
            for j in deps_idx:
                for output in task_info_lst[j]['outputs']:
                    if output in task_info['inputs']:
                        defined_inputs.append(output)

        task_info['schema'].check_missing_inputs(defined_inputs)


def get_dependency_idx(task_info_lst):

    dependency_idx = []
    all_outputs = []
    for task_info in task_info_lst:
        all_outputs.extend(task_info['outputs'])
        output_idx = []
        for input_j in task_info['inputs']:
            for task_idx_k, task_info_k in enumerate(task_info_lst):
                if input_j in task_info_k['outputs']:
                    output_idx.append(task_idx_k)
        else:
            dependency_idx.append(output_idx)

    if len(all_outputs) != len(set(all_outputs)):
        msg = 'Multiple tasks in the workflow have the same output!'
        raise IncompatibleWorkflow(msg)

    # Check for circular dependencies in task inputs/outputs:
    all_deps = []
    for idx, deps in enumerate(dependency_idx):
        for i in deps:
            all_deps.append(tuple(sorted([idx, i])))

    if len(all_deps) != len(set(all_deps)):
        msg = 'Workflow tasks are circularly dependent!'
        raise IncompatibleWorkflow(msg)

    return dependency_idx


def get_task_num_elements(downstream_task, upstream_tasks):

    num_elements = downstream_task['length']

    if upstream_tasks:

        is_nesting_mixed = len(set([i['nest'] for i in upstream_tasks])) > 1

        for i in upstream_tasks:

            if i['merge_priority'] is None and is_nesting_mixed:
                msg = ('`merge_priority` for task "{}" must be specified, because'
                       ' nesting is mixed.')
                raise MissingMergePriority(msg.format(i['name']))

            elif i['merge_priority'] is not None and not is_nesting_mixed:
                msg = ('`merge_priority` value is specified but not required for '
                       'task "{}", because nesting is not mixed.')
                warn(msg.format(downstream_task['name']))

        all_merge_priority = [i.get('merge_priority') or 0 for i in upstream_tasks]
        merging_order = np.argsort(all_merge_priority)

        for i in merging_order:
            task_to_merge = upstream_tasks[i]
            if task_to_merge['nest']:
                num_elements *= task_to_merge['num_elements']
            else:
                if task_to_merge['num_elements'] != num_elements:
                    msg = ('Cannot merge without nesting task "{}" (with {} elements)'
                           ' with task "{}" (with {} elements [during '
                           'merge]).'.format(
                               task_to_merge['name'],
                               task_to_merge['num_elements'],
                               downstream_task['name'],
                               num_elements,
                           ))
                    raise IncompatibleTaskNesting(msg)

    return num_elements


def get_task_elements_idx(downstream_task, upstream_tasks):
    """
    Get the elements indices of upstream task outputs (and those of downstream task local
    inputs) necessary to prepare element inputs for the downstream task.

    Parameters
    ----------
    downstream_task : dict
    upstream_tasks : list of dict

    Returns
    -------
    task_elements_idx : dict of (int: list)
        Dict whose keys are task indices (`task_idx`) and whose values are 1D lists
        corresponding to the element indices from upstream task outputs (or the 
        downstream task local inputs) for each task.

    """

    task_elements_idx = {}

    # First find elements indices for downstream local inputs:
    inputs_idx = np.arange(downstream_task['length'])
    tile_num = downstream_task['length']
    for j in upstream_tasks:
        rep_num = j['num_elements'] if j.get('nest') else 1
        inputs_idx = np.repeat(inputs_idx, rep_num)

    task_elements_idx.update({downstream_task['task_idx']: list(inputs_idx)})

    # Now find element indices for upstream task outputs:
    for idx, i in enumerate(upstream_tasks):

        tile_num_i = 1
        if i.get('nest'):
            tile_num_i = tile_num
            tile_num *= i['num_elements']

        inputs_idx = np.tile(np.arange(i['num_elements']), tile_num_i)

        for j in upstream_tasks[idx + 1:]:
            rep_num = j['num_elements'] if j.get('nest') else 1
            inputs_idx = np.repeat(inputs_idx, rep_num)

        task_elements_idx.update({i['task_idx']: list(inputs_idx)})

    return task_elements_idx


def get_input_elements_idx(task_elements_idx, downstream_task, task_info_lst):

    params_idx = {}
    for input_name in downstream_task['inputs']:
        # Find the task_idx for which this input is an output:
        input_task_idx = None
        for i in task_info_lst:
            if input_name in i['outputs']:
                input_task_idx = i['task_idx']
                param_task_idx = input_task_idx
                break
        if input_task_idx is None:
            input_task_idx = downstream_task['task_idx']
            params_idx.update({
                input_name: {
                    'input_idx': task_elements_idx[input_task_idx],
                }
            })
        else:
            params_idx.update({
                input_name: {
                    'task_idx': param_task_idx,
                    'output_idx': task_elements_idx[input_task_idx],
                }
            })

    return params_idx
