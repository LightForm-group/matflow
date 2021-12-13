"""matflow.models.workflow.py

Module containing the Workflow class and some functions used to decorate Workflow methods.

"""

import copy
import functools
import re
import secrets
import pickle
import shutil
from datetime import datetime
from enum import Enum
from pathlib import Path
from warnings import warn

import h5py
import hickle
import hpcflow
import numpy as np
from ruamel.yaml import YAML, scalarstring

from matflow import __version__
from matflow.config import Config
from matflow.errors import (
    WorkflowPersistenceError,
    TaskElementExecutionError,
    UnexpectedSourceMapReturnError,
    WorkflowIterationError,
    ParameterImportError,
)
from matflow.hicklable import to_hicklable
from matflow.models.command import DEFAULT_FORMATTERS
from matflow.models.construction import init_tasks, get_element_idx
from matflow.models.software import SoftwareInstance
from matflow.models.task import TaskStatus, DEFAULT_TASK_CONTEXT, Task
from matflow.models.parameters import Parameters
from matflow.utils import (
    parse_times,
    zeropad,
    datetime_to_dict,
    get_nested_item,
    nested_dict_arrays_to_list,
    index,
    working_directory,
)


def requires_ids(func):
    """Workflow method decorator to raise if IDs are not assigned."""
    @functools.wraps(func)
    def func_wrap(self, *args, **kwargs):
        if not self.id:
            raise ValueError('Run `set_ids()` before using this method.')
        return func(self, *args, **kwargs)
    return func_wrap


def requires_path_exists(func):
    """Workflow method decorator to raise if workflow path does not exist as a directory."""
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
        '_dependency_idx',
        '_history',
        '_archives',
        '_archive_excludes',
        '_figures',
        '_metadata',
        '_num_iterations',
        '_iterate',
        '_iterate_run_options',
        '_import_list',
        '_imported_parameters',
    ]

    def __init__(self, name, tasks, stage_directory=None, extends=None, archives=None,
                 archive_excludes=None, figures=None, metadata=None, num_iterations=None,
                 iterate=None, iterate_run_options=None, import_list=None,
                 imported_parameters=None, check_integrity=True, profile=None,
                 __is_from_file=False):

        self._id = None             # Assigned once by set_ids()
        self._human_id = None       # Assigned once by set_ids()
        self._loaded_path = None    # Assigned on save or on load.

        self._is_from_file = __is_from_file
        self._name = name
        self._extends = [str(Path(i).resolve()) for i in (extends or [])]
        self._stage_directory = str(Path(stage_directory or '').resolve())
        self._profile = profile
        self._archives = archives or []
        self._archive_excludes = archive_excludes
        self._figures = [{'idx': idx, **i}
                         for idx, i in enumerate(figures)
                         ] if figures else []
        self._metadata = metadata or {}

        self._import_list = self._validate_import_list(import_list, self.is_from_file)
        self._imported_parameters = imported_parameters or {}

        if not self.is_from_file and self.import_list:
            for import_item in self.import_list:
                imported_data = self.import_parameter(
                    parameter_name=import_item['parameter'],
                    workflow_path=import_item['from']['workflow'],
                    context=import_item['from'].get('context'),
                    task_idx=import_item['from'].get('task_idx'),
                    iteration_idx=import_item['from'].get('iteration'),
                    elements_idx=import_item['from'].get('elements'),
                    new_context=import_item.get('context'),
                    new_parameter_name=import_item.get('as'),
                )
                param_name = imported_data.pop('parameter_name')
                self._imported_parameters.update({
                    (param_name, imported_data['context']): imported_data
                })

        tasks, task_elements, dep_idx = init_tasks(
            self,
            tasks,
            self.imported_parameters,
            self.is_from_file,
            check_integrity,
        )
        self._tasks = tasks
        self._dependency_idx = dep_idx

        self._num_iterations = num_iterations or 1
        self._iterate = self._validate_iterate(iterate, self.is_from_file)
        self._iterate_run_options = iterate_run_options or {}

        # Find element indices that determine the elements from which task inputs are
        # drawn:
        task_lst = [
            {
                'task_idx': i.task_idx,
                'local_inputs': i.local_inputs,
                'name': i.name,
                'schema': i.schema,
                'context': i.context,
            } for i in tasks
        ]
        elements_idx = get_element_idx(
            task_lst,
            dep_idx,
            self.num_iterations,
            self.iterate,
            self.imported_parameters,
        )

        for task in self.tasks:
            if self.is_from_file:
                elements = task_elements[task.task_idx]
            else:
                num_elements = elements_idx[task.task_idx]['num_elements']
                elements = [{'element_idx': elem_idx} for elem_idx in range(num_elements)]
            task.init_elements(elements)

        self._elements_idx = elements_idx

        if not self.is_from_file:
            self._check_archive_connection()
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
            ) + (
                f'    Context: {i.context}\n' if i.context != DEFAULT_TASK_CONTEXT else ''
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

    def _check_archive_connection(self):
        for archive in self.archives:
            if not archive:
                return

            # TODO: should this whole function be outsourced to hpcflow?

            # TODO: check local archive path exists

            if 'cloud_provider' in self.archive_definitions[archive]:
                provider = self.archive_definitions[archive]['cloud_provider']
                hpcflow.cloud_connect(
                    provider, config_dir=Config.get('hpcflow_config_dir'))

            # TODO when supported in DataLight, add datalight.check_access(...)

    def _append_history(self, action, **kwargs):
        """Append a new history event."""

        if action not in WorkflowAction:
            raise TypeError('`action` must be a `WorkflowAction`.')

        new_hist = {
            'action': action,
            'matflow_version': __version__,
            'timestamp': datetime.now(),
            'action_info': kwargs,
        }
        self._history.append(new_hist)

        if action is not WorkflowAction.generate:

            # Make hickle-able:
            new_hist = copy.deepcopy(new_hist)
            new_hist['action'] = (new_hist['action'].name, new_hist['action'].value)
            new_hist['timestamp'] = datetime_to_dict(new_hist['timestamp'])

            with h5py.File(self.loaded_path, 'r+') as handle:

                # Load and save attributes of history list:
                path = self.HDF5_path + "/'history'"
                attributes = dict(handle[path].attrs)
                history = hickle.load(handle, path=path)
                del handle[path]

                # Append to and re-dump history list:
                history.append(new_hist)
                hickle.dump(history, handle, path=path)

                # Update history list attributes to maintain /workflow_obj loadability
                for k, v in attributes.items():
                    handle[path].attrs[k] = v

    def _validate_import_list(self, import_list, is_from_file):

        if not import_list:
            return ()

        if not isinstance(import_list, (list, tuple)):
            raise ValueError(f'`import_list` must be a list or tuple.')

        req_keys = {'parameter', 'from'}
        good_keys = req_keys | {'as', 'context'}
        from_req_keys = {'workflow'}
        from_good_keys = from_req_keys | {'context', 'iteration', 'elements', 'task_idx'}

        for idx, import_item in enumerate(import_list):

            if not isinstance(import_item, dict):
                msg = f'Each item in `import_list` must be a dict, but item {idx} is not.'
                raise ValueError(msg)

            import_item_keys = set(import_item)
            miss_keys = req_keys - import_item_keys
            bad_keys = import_item_keys - good_keys
            msg = (f'Each items in `import_list` must be a dict with particular keys. '
                   f'For item {idx}, ')
            if miss_keys:
                miss_keys_fmt = ', '.join([f'"{i}"' for i in miss_keys])
                raise ValueError(msg + f'missing keys are: {miss_keys_fmt}.')
            if bad_keys:
                bad_keys_fmt = ', '.join([f'"{i}"' for i in bad_keys])
                raise ValueError(msg + f'unknown keys are: {bad_keys_fmt}.')

            if not import_item.get('as'):
                import_item['as'] = import_item['parameter']  # set default import alias

            if not isinstance(import_item['from'], dict):
                msg = (f'The `from` option of import item {idx} of `import_list` must be '
                       f'a dict, but it is not.')
                raise ValueError(msg)

            new_context = import_item.get('context')
            if new_context:
                import_item['context'] = Task.make_safe_context(new_context)

            from_item_keys = set(import_item['from'])
            from_miss_keys = from_req_keys - from_item_keys
            from_bad_keys = from_item_keys - from_good_keys
            msg = (f'The `from` option of import item {idx} of `import_list` must be '
                   f'a dict with particular keys. ')
            if from_miss_keys:
                from_miss_keys_fmt = ', '.join([f'"{i}"' for i in from_miss_keys])
                raise ValueError(msg + f'Missing keys are: {from_miss_keys_fmt}.')
            if from_bad_keys:
                from_bad_keys_fmt = ', '.join([f'"{i}"' for i in from_bad_keys])
                raise ValueError(msg + f'Unknown keys are: {from_bad_keys_fmt}.')

            from_context = import_item['from'].get('context')
            if from_context:
                import_item['from']['context'] = Task.make_safe_context(from_context)

            if not is_from_file:
                # Check workflow is a file:
                workflow_path = Path(import_item['from']['workflow']).resolve()
                if not workflow_path.is_file():
                    trial_workflow_path = workflow_path / 'workflow.hdf5'
                    if not trial_workflow_path.is_file():
                        msg = (f'The workflow path specified in import item {idx} of '
                               f'`import_list` is not a file (nor is this path joined '
                               f'with "workflow.hdf5"): "{workflow_path}".')
                        raise ValueError(msg)
                    else:
                        workflow_path = trial_workflow_path

                import_item['from']['workflow'] = str(workflow_path)

        return tuple(import_list)

    def _validate_iterate(self, iterate_dict, is_from_file):

        if not iterate_dict:
            return iterate_dict

        elif self.num_iterations != 1:
            msg = "Specify either `iterate` (dict) or `num_iterations` (int)."
            raise ValueError(msg)

        req_keys = ['parameter', 'num_iterations']
        if is_from_file:
            req_keys.extend(['task_pathway', 'producing_task', 'originating_tasks'])
        allowed_keys = set(req_keys)

        miss_keys = list(set(req_keys) - set(iterate_dict))
        bad_keys = list(set(iterate_dict) - allowed_keys)
        msg = '`iterate` must be a dict.'
        if miss_keys:
            miss_keys_fmt = ', '.join(['"{}"'.format(i) for i in miss_keys])
            raise WorkflowIterationError(msg + f' Missing keys are: {miss_keys_fmt}.')
        if bad_keys:
            bad_keys_fmt = ', '.join(['"{}"'.format(i) for i in bad_keys])
            raise WorkflowIterationError(msg + f' Unknown keys are: {bad_keys_fmt}.')

        if not is_from_file:
            task_pathway = self.get_iteration_task_pathway(iterate_dict['parameter'])
            iterate_dict.update(task_pathway)

        return iterate_dict

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
        """Get name without spaces"""
        return self.name.replace(' ', '_')

    @property
    def name_friendly(self):
        """Capitalise and remove underscores"""
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
    def figures(self):
        return self._figures

    @property
    def metadata(self):
        return self._metadata

    @property
    def num_iterations(self):
        return self._num_iterations

    @property
    def iterate(self):
        return self._iterate

    @property
    def iterate_run_options(self):
        return self._iterate_run_options

    @property
    def elements_idx(self):
        return self._elements_idx

    @property
    def dependency_idx(self):
        return self._dependency_idx

    @property
    def extends(self):
        return [Path(i) for i in self._extends]

    @property
    def archives(self):
        return self._archives

    @property
    def archive_excludes(self):
        schema_excludes = [
            i
            for task in self.tasks
            for i in task.schema.archive_excludes or []
        ]
        return list(set(self._archive_excludes or [] + schema_excludes))

    @property
    def archive_definitions(self):
        if not self.archives:
            return None
        else:
            archive_defns = {}
            for archive in self.archives:
                archive_defn = {
                    **Config.get('archive_locations')[archive],
                    'root_directory_name': 'parent',
                }
                archive_defns.update({archive: archive_defn})
        return archive_defns

    @property
    def stage_directory(self):
        return Path(self._stage_directory)

    @property
    def path_exists(self):
        """Does the Workflow project directory exist on this machine?"""
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
        """Get the full path of the Workflow project as a Path."""
        return Path(self.stage_directory, self.human_id)

    @property
    def path_str(self):
        """Get the full path of the Workflow project as a string."""
        return str(self.path)

    @property
    def default_file_path(self):
        return self.path.joinpath('workflow.hdf5')

    @property
    def HDF5_path(self):
        return '/workflow_obj/data'

    @property
    def import_list(self):
        return self._import_list

    @property
    def imported_parameters(self):
        return self._imported_parameters

    def import_parameter(self, parameter_name, workflow_path, context=None, task_idx=None,
                         iteration_idx=None, elements_idx=None, new_context=None,
                         new_parameter_name=None):
        """
        Import an output parameter from another workflow.

        Parameters
        ----------
        parameter_name : str
            Name of parameter to import.
        workflow_path : str or Path
            Full path to the workflow HDF5 file from which the parameter should be
            imported.
        context : str, optional
            The task context from which the parameter should be imported. If not
            specified, the default context will be used.
        task_idx : int, optional
            In the case of a workflow with parameter-modifying tasks, multiple tasks with
            the same context may output the same parameter. The index of the specific task
            from which the parameter should be imported can be specified here. If this is
            given, the `context` argument will be ignored.
        iteration_idx : int, optional
            For parmaters that have been output from tasks that have multiple iterations,
            a specific iteration index must be chosen, from which the parameter will be
            imported. If not specified, the final iteration (index -1) will be used.
        elements_idx : (list or tuple) of int, optional
            The indices of the elements from which to import the value of this parameter.
            Note that these should be specified relative to the specified iteration. So
            for a task with four elements per iteration and N iterations, the allowable
            values of `elements_idx` would be 0-3.
        new_context : str, optional
            The new context to use for the imported parameter.
        new_parameter_name : str, optional
            The new name to use for the imported parameter.

        Returns
        -------
        imported_data_dict : dict

        """

        if not new_parameter_name:
            new_parameter_name = parameter_name

        if new_context is None:
            new_context = context or DEFAULT_TASK_CONTEXT

        imp_workflow = Workflow.load_HDF5_file(workflow_path, full_path=True)

        if task_idx is not None:
            # If given directly, `task_idx` overrides specified context:

            if task_idx in imp_workflow.get_output_tasks(parameter_name):
                import_task_idx = task_idx

            elif task_idx not in range(len(imp_workflow.tasks)):
                msg = (f'The parameter "{parameter_name}" cannot be imported from task '
                       f'index {task_idx} because no such task exists.')
                raise ParameterImportError(msg)

            else:
                ctx = imp_workflow.tasks[task_idx].context
                context_msg = (f' and context "{ctx}"'
                               if ctx is not DEFAULT_TASK_CONTEXT else '')
                msg = (
                    f'The parameter "{parameter_name}" cannot be imported from task '
                    f'index {task_idx} with name "{imp_workflow.tasks[task_idx].name}"'
                    f'{context_msg}, since this task does not output this parameter.'
                )
                raise ParameterImportError(msg)

        else:
            # Use context to get the correct task:
            output_task_idx = imp_workflow.get_output_tasks(parameter_name, context)

            if not output_task_idx:
                context_msg = f' with context "{context}"' if context is not None else ''
                msg = (
                    f'The parameter "{parameter_name}"{context_msg} cannot be found as an'
                    f' output in any of the tasks in the specified workflow: '
                    f'{workflow_path}.'
                )
                raise ParameterImportError(msg)

            elif len(output_task_idx) == 1:
                import_task_idx = output_task_idx[0]

            else:
                # Could be due to multiple contexts (and context not specified) and/or
                # parameter-modifying task(s) (and task_idx not specified)

                all_contexts = [i.context for i in imp_workflow.tasks
                                if i.task_idx in output_task_idx]
                is_multi_context = len(set(all_contexts)) > 1
                if is_multi_context:
                    multi_contexts_fmt = ', '.join([f'"{i}"' for i in all_contexts])
                    msg = (
                        f'The parameter "{parameter_name}" exists as an output for tasks '
                        f'with multiple contexts: {multi_contexts_fmt}. Specify from '
                        f'which context you wish to import this parameter.'
                    )
                    raise ParameterImportError(msg)
                else:
                    # Use the task with the highest task_idx:
                    import_task_idx = max(output_task_idx)

        imp_task = imp_workflow.tasks[import_task_idx]

        # Set a default iteration index (final iteration):
        final_iter_idx = max(imp_task.elements_idx['iteration_idx'])
        if iteration_idx is None:
            iteration_idx = final_iter_idx
        elif iteration_idx < 0:
            iteration_idx += imp_task.elements_idx['num_iterations']

        # Get element indices within the specified iteration:
        num_elems_per_iter = imp_task.elements_idx['num_elements_per_iteration']
        if elements_idx:
            if not isinstance(elements_idx, (list, tuple)):
                msg = (f'The `elements_idx` option for importing parameter '
                       f'"{parameter_name}" must be a list of element indices.')
                raise ParameterImportError(msg)
            in_range = all([i in range(num_elems_per_iter) for i in elements_idx])
            if not in_range:
                msg = (f'The `elements_idx` option for importing parameter '
                       f'"{parameter_name}" must be a list of element indices. The '
                       f'allowable indices are: 0 to {num_elems_per_iter - 1} '
                       f'(inclusive).')
        else:
            elements_idx = range(num_elems_per_iter)  # all elements in iteration

        # Get Element objects:
        abs_elements_idx = [i + (num_elems_per_iter * iteration_idx)
                            for i in elements_idx]
        element_objs = [i for i in imp_task.elements if i.element_idx in abs_elements_idx]

        new_groups = self.filter_imported_parameter_groups(
            parameter_name,
            imp_task.elements_idx['groups'],
            elements_idx,
        )
        imported_data_dict = {
            'parameter_name': new_parameter_name,
            'original_name': parameter_name,
            'context': new_context,
            'groups': new_groups,
            'data': [],             # populated below, deleted on self.write_HDF5_file
            'data_idx': [],         # populated in self.write_HDF5_file
            'iteration_idx': [],    # populated below
        }

        for element in element_objs:
            try:
                data = element.get_output(parameter_name)
            except (KeyError, AttributeError):
                msg = (f'The output parameter "{parameter_name}" from element '
                       f'{element.element_idx} of task "{imp_task.unique_name}" could '
                       f'not be found in the workflow at "{workflow_path}".')
                raise ParameterImportError(msg)
            imported_data_dict['data'].append(data)
            imported_data_dict['iteration_idx'].append(iteration_idx)

        return imported_data_dict

    @staticmethod
    def filter_imported_parameter_groups(parameter_name, original_groups, elements_idx):
        """Filter groups of imported paramaters according to the requested subset of
        elements to include in the importation."""

        new_groups = copy.deepcopy(original_groups)
        modifiable_dat_keys = [
            'group_idx_per_iteration',
            'group_element_idx_per_iteration',
            'num_groups_per_iteration',
            'group_size_per_iteration',
            'num_groups',
        ]
        for group_name, group_dat in new_groups.items():

            group_dat['pre_import'] = {}
            # Copy original keys to a separate dict:
            for key in modifiable_dat_keys:
                group_dat['pre_import'].update({key: copy.deepcopy(group_dat[key])})

            # Remove elements not in elements_idx:
            new_group_idx = [
                i for idx, i in enumerate(group_dat['group_idx_per_iteration'])
                if idx in elements_idx
            ]

            # Remap group indices consecutively starting from zero:
            _, new_group_idx = np.unique(new_group_idx, return_inverse=True)

            # Get the the inverse to update `group_element_idx_per_iteration`:
            new_group_elem_idx = [
                list(np.where(np.array(new_group_idx) == i)[0])
                for i in range(max(new_group_idx) + 1)
            ]

            # Check for ragged group sizes:
            uniq_group_sizes = set([len(i) for i in new_group_elem_idx])
            if len(uniq_group_sizes) > 1:
                grp_nm_fmt = group_name.split('user_group_')[1]
                msg = (
                    f'The requested subset of elements to import for parameter '
                    f'"{parameter_name}" ({list(elements_idx)}) has resulted in a ragged '
                    f'group "{grp_nm_fmt}", where not all "{grp_nm_fmt}" groups have the '
                    f'same size. Distinct group sizes are: {list(uniq_group_sizes)}. '
                    f'Original group size was: '
                    f'{group_dat["pre_import"]["group_size_per_iteration"]}. The element '
                    f'subset must be selected such that ragged group sizes are not '
                    f'created.'
                )
                raise ParameterImportError(msg)

            group_dat['group_idx_per_iteration'] = list(new_group_idx)
            group_dat['group_element_idx_per_iteration'] = list(new_group_elem_idx)
            group_dat['num_groups_per_iteration'] = len(new_group_elem_idx)
            group_dat['group_size_per_iteration'] = len(new_group_elem_idx[0])

            # Only a single iteration can be imported:
            group_dat['num_groups'] = group_dat['num_groups_per_iteration']

        return new_groups

    @functools.lru_cache()
    def get_element_data(self, idx):

        with h5py.File(self.loaded_path, 'r') as handle:

            path = f'/element_data'
            num_dat = len(handle[path])
            is_list = True if isinstance(idx, tuple) else False
            if not is_list:
                idx = [idx]

            # Element data are zero padded and include name of parameter for convenience,
            # so map their integer index to the actual group names:
            idx_map = {int(re.search(r'(\d+)', i).group()): i for i in handle[path]}

            out = []
            for i in idx:
                if i > (num_dat - 1):
                    warn(f'Element data has {num_dat} member(s), but idx={i} '
                         f'requested. Perhaps some element data has been deleted.')
                dat_path = path + f'/{idx_map[i]}'
                out.append(hickle.load(handle, path=dat_path))

            if not is_list:
                out = out[0]
            return out

    def get_task_idx_padded(self, task_idx, ret_zero_based=True):
        """Get a task index, zero-padded according to the number of tasks."""
        if ret_zero_based:
            return zeropad(task_idx, len(self) - 1)
        else:
            return zeropad(task_idx + 1, len(self))

    @requires_path_exists
    def get_task_path(self, task_idx):
        """Get the path to a task directory."""
        if task_idx > (len(self) - 1):
            msg = f'Workflow has only {len(self)} tasks.'
            raise ValueError(msg)
        task = self.tasks[task_idx]
        task_idx_fmt = self.get_task_idx_padded(task_idx, ret_zero_based=False)
        task_path = self.path.joinpath(f'task_{task_idx_fmt}_{task.unique_name}')
        return task_path

    @requires_path_exists
    def get_task_sources_path(self, task_idx):
        task_path = self.get_task_path(task_idx)
        return task_path.joinpath('sources')

    @requires_path_exists
    def get_element_path(self, task_idx, element_idx):
        """Get the path to an element directory."""
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

    @requires_path_exists
    def _get_element_temp_array_prepare_path(self, task_idx, element_idx):
        task = self.tasks[task_idx]
        element_path = self.get_element_path(task_idx, element_idx)
        out = element_path.joinpath(f'task_prepare_{task.id}_element_{element_idx}.hdf5')
        return out

    @requires_path_exists
    def _get_element_temp_array_process_path(self, task_idx, element_idx):
        task = self.tasks[task_idx]
        element_path = self.get_element_path(task_idx, element_idx)
        out = element_path.joinpath(f'task_process_{task.id}_element_{element_idx}.hdf5')
        return out

    def write_element_directories(self, iteration_idx):
        """Generate element directories for a given iteration."""

        for elems_idx, task in zip(self.elements_idx, self.tasks):

            if (
                iteration_idx > 0 and
                self.iterate and
                task.task_idx not in self.iterate['task_pathway']
            ):
                continue

            task_idx = task.task_idx
            num_elems = elems_idx['num_elements_per_iteration']
            iter_elem_idx = [i + (iteration_idx * num_elems) for i in range(num_elems)]

            # Generate element directories:
            for elem_idx_i in iter_elem_idx:
                self.get_element_path(task_idx, elem_idx_i).mkdir(exist_ok=True)

            # Copy any local input files to the element directories:
            for input_alias, inputs_idx in self.elements_idx[task_idx]['inputs'].items():

                schema_input = [i for i in task.schema.inputs
                                if i['alias'] == input_alias][0]

                if schema_input['file'] == False:
                    continue

                input_name = schema_input['name']
                input_dict = task.local_inputs['inputs'][input_name]
                local_ins = [input_dict['vals'][i] for i in input_dict['vals_idx']]

                for elem_idx_i in iter_elem_idx:

                    # Path as originally specified (absolute/relative/maybe with `~` etc):
                    file_path = local_ins[inputs_idx['local_input_idx'][elem_idx_i]]
                    file_path = Path(file_path).expanduser()

                    # If `file_path` is absolute, joinpath will simply return `file_path`:
                    file_path_full = self.stage_directory.joinpath(file_path)
                    elem_path = self.get_element_path(task_idx, elem_idx_i)
                    dst_path = elem_path.joinpath(file_path_full.name)

                    if not file_path_full.is_file():
                        msg = (f'Input file "{input_name}" with path "{file_path_full}" '
                               f'does not exist!')
                        raise ValueError(msg)

                    shutil.copyfile(file_path_full, dst_path)

                    if schema_input['file'] and schema_input['save']:
                        # Save file as string in workflow:
                        with file_path_full.open('r') as handle:
                            file_dat = handle.read()
                        element = task.elements[elem_idx_i]
                        element.add_file(file_path_full.name, value=file_dat)

    def prepare_iteration(self, iteration_idx):

        for elems_idx, task in zip(self.elements_idx, self.tasks):

            if (
                iteration_idx > 0 and
                self.iterate and
                task.task_idx not in self.iterate['task_pathway']
            ):
                continue

            num_elems = elems_idx['num_elements_per_iteration']
            iter_elem_idx = [i + (iteration_idx * num_elems) for i in range(num_elems)]
            cmd_line_inputs, input_vars = self._get_command_line_inputs(task.task_idx)

            for local_in_name, var_name in input_vars.items():

                var_file_name = '{}.txt'.format(var_name)

                # Create text file in each element directory for each in `input_vars`:
                for elem_idx_i in iter_elem_idx:

                    task_elem_path = self.get_element_path(task.task_idx, elem_idx_i)
                    in_val = cmd_line_inputs[local_in_name][elem_idx_i]

                    var_file_path = task_elem_path.joinpath(var_file_name)
                    with var_file_path.open('w') as handle:
                        handle.write(in_val + '\n')

    def write_directories(self):
        """Generate task and element directories for the first iteration."""

        for task in self.tasks:

            # Generate task directory:
            self.get_task_path(task.task_idx).mkdir()

            if task.software_instance.requires_sources:
                self.get_task_sources_path(task.task_idx).mkdir()

        self.write_element_directories(iteration_idx=0)

    def get_hpcflow_job_name(self, task, job_type, is_stats=False):
        """Get the scheduler job name for a given task index and job type.

        Parameters
        ----------
        task : Task
        job_type : str
            One of "prepare-task", "process-task", "process-prepare-task", "run",
            "prepare-sources". If "process-prepare-task", the task passed must be the
            "processing" task.
        is_stats : bool, optional

        Returns
        -------
        job_name : str
            The job name to be used in the hpcflow workflow.

        """
        ALLOWED = ['prepare-task', 'process-task', 'process-prepare-task', 'run',
                   'prepare-sources']
        if job_type not in ALLOWED:
            raise ValueError(f'Invalid `job_type`. Allowed values are: {ALLOWED}.')

        task_idx_fmt = self.get_task_idx_padded(task.task_idx, ret_zero_based=False)

        base = 't' if not is_stats else 's'

        if job_type == 'run':
            out = f'{base}{task_idx_fmt}'

        elif job_type in ['prepare-task', 'process-task']:
            out = f'{base}{task_idx_fmt}_{job_type[:3]}'

        elif job_type == 'process-prepare-task':
            next_task = self.tasks[task.task_idx + 1]
            next_task_idx_fmt = self.get_task_idx_padded(
                next_task.task_idx,
                ret_zero_based=False,
            )
            out = f'{base}{task_idx_fmt}+{next_task_idx_fmt}_aux'

        elif job_type == 'prepare-sources':
            out = f'{base}{task_idx_fmt}_src'

        return out

    def _get_command_line_inputs(self, task_idx):

        task = self.tasks[task_idx]
        elems_idx = self.elements_idx[task_idx]
        _, input_vars = task.get_formatted_commands()

        cmd_line_inputs = {}
        for local_in_name, local_in in task.local_inputs['inputs'].items():

            if local_in_name in input_vars:
                # TODO: We currently only consider input_vars for local inputs.

                # Expand values for intra-task nesting:
                values = [self.get_element_data(i)
                          for i in local_in['vals_data_idx']]

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
                    values_fmt[i] if i is not None else None
                    for i in elems_idx['inputs'][local_in_name]['local_input_idx']
                ]
                cmd_line_inputs.update({local_in_name: values_fmt_all})

        return cmd_line_inputs, input_vars

    @requires_path_exists
    def get_hpcflow_workflow(self):
        """Generate an hpcflow workflow to execute this workflow."""

        command_groups = []
        variables = {}
        for elems_idx, task in zip(self.elements_idx, self.tasks):

            src_prep_cmd_group = []
            executable = task.software_instance.executable

            if task.software_instance.requires_sources:

                sources_prep = task.software_instance.sources_preparation
                sources_dir = str(self.get_task_sources_path(task.task_idx))
                prep_env = sources_prep.env.as_str()

                source_map = Config.get('sources_maps')[
                    (task.name, task.method, task.software)
                ]
                src_vars = source_map['sources']
                prep_cmds = sources_prep.get_formatted_commands(
                    src_vars,
                    sources_dir,
                    task.task_idx,
                )

                for src_var_name, src_name in src_vars.items():
                    executable = executable.replace(f'<<{src_var_name}>>', src_name)
                    executable = executable.replace('<<sources_dir>>', sources_dir)

                src_prep_cmd_group = [{
                    'directory': str(self.get_task_sources_path(task.task_idx)),
                    'nesting': 'hold',
                    'commands': prep_cmds,
                    'environment': prep_env,
                    'stats': False,
                    'scheduler_options': task.get_scheduler_options('prepare'),
                    'name': self.get_hpcflow_job_name(task, 'prepare-sources'),
                    'meta': {'from_tasks': [task.task_idx]},
                }]

            if task.schema.is_func:
                # The task is to be run directly in Python:
                # (SGE specific)
                fmt_commands = [
                    {
                        'line': (f'matflow run-python-task --task-idx={task.task_idx} '
                                 f'--element-idx='
                                 f'$((($ITER_IDX * $SGE_TASK_LAST) + $SGE_TASK_ID - 1)) '
                                 f'--directory={self.path}')
                    }
                ]

            else:

                fmt_commands, input_vars = task.get_formatted_commands()
                # `input_vars` are those inputs that appear directly in the commands:

                fmt_commands_new = []
                for i in fmt_commands:
                    i['line'] = i['line'].replace('<<executable>>', executable)
                    fmt_commands_new.append(i)
                fmt_commands = fmt_commands_new

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

            task_path_rel = str(self.get_task_path(task.task_idx).name)

            cur_prepare_opts = task.get_scheduler_options('prepare')
            if task.task_idx == 0:
                prev_task, prev_process_opts = None, None
            else:
                prev_task = self.tasks[task.task_idx - 1]
                prev_process_opts = prev_task.get_scheduler_options('process')

            if (
                task.task_idx == 0 or (
                    cur_prepare_opts != prev_process_opts or
                    'job_array' in task.prepare_run_options or
                    'job_array' in prev_task.process_run_options
                )
            ):
                if 'job_array' in task.prepare_run_options:
                    command_groups.extend([
                        {
                            'directory': '<<{}_dirs>>'.format(task_path_rel),
                            'nesting': 'hold',
                            'commands': task.get_prepare_task_element_commands(
                                is_array=True
                            ),
                            'stats': False,
                            'scheduler_options': cur_prepare_opts,
                            'name': self.get_hpcflow_job_name(task, 'prepare-task'),
                            'meta': {'from_tasks': [task.task_idx]},
                        },
                        {
                            'directory': '.',
                            'nesting': 'hold',
                            'commands': task.get_prepare_task_commands(is_array=True),
                            'stats': False,
                            'scheduler_options': cur_prepare_opts,
                            'name': self.get_hpcflow_job_name(task, 'prepare-task'),
                            'meta': {'from_tasks': [task.task_idx]},
                        }
                    ])

                else:
                    command_groups.append({
                        'directory': '.',
                        'nesting': 'hold',
                        'commands': task.get_prepare_task_commands(is_array=False),
                        'stats': False,
                        'scheduler_options': cur_prepare_opts,
                        'name': self.get_hpcflow_job_name(task, 'prepare-task'),
                        'meta': {'from_tasks': [task.task_idx]},
                    })

            if task.software_instance.requires_sources:
                command_groups += src_prep_cmd_group

            main_task = {
                'directory': '<<{}_dirs>>'.format(task_path_rel),
                'nesting': 'hold',
                'commands': fmt_commands,
                'scheduler_options': task.get_scheduler_options('main'),
                'name': self.get_hpcflow_job_name(task, 'run'),
                'stats': task.stats,
                'stats_name': self.get_hpcflow_job_name(task, 'run', is_stats=True),
                'meta': {'from_tasks': [task.task_idx]},
            }
            env = task.software_instance.env.as_str()
            if env:
                main_task.update({'environment': env})
            alt_scratch = task.run_options.get('alternate_scratch')
            if alt_scratch:
                main_task.update({'alternate_scratch': alt_scratch})

            command_groups.append(main_task)

            add_process_groups = True
            cur_process_opts = task.get_scheduler_options('process')
            if task.task_idx < (len(self) - 1):
                next_task = self.tasks[task.task_idx + 1]
                next_prepare_opts = next_task.get_scheduler_options('prepare')
                if (
                    cur_process_opts == next_prepare_opts and
                    'job_array' not in task.process_run_options and
                    'job_array' not in next_task.prepare_run_options
                ):
                    # Combine into one command group:
                    command_groups.append({
                        'directory': '.',
                        'nesting': 'hold',
                        'commands': (
                            task.get_process_task_commands(is_array=False) +
                            next_task.get_prepare_task_commands(is_array=False)
                        ),
                        'stats': False,
                        'scheduler_options': cur_process_opts,
                        'name': self.get_hpcflow_job_name(task, 'process-prepare-task'),
                        'meta': {'from_tasks': [task.task_idx, next_task.task_idx]},
                    })
                    add_process_groups = False

            if add_process_groups:
                if 'job_array' in task.process_run_options:
                    command_groups.extend([
                        {
                            'directory': '<<{}_dirs>>'.format(task_path_rel),
                            'nesting': None,
                            'commands': task.get_process_task_element_commands(
                                is_array=True
                            ),
                            'stats': False,
                            'scheduler_options': cur_process_opts,
                            'name': self.get_hpcflow_job_name(task, 'process-task'),
                            'meta': {'from_tasks': [task.task_idx]},
                        },
                        {
                            'directory': '.',
                            'nesting': 'hold',
                            'commands': task.get_process_task_commands(is_array=True),
                            'stats': False,
                            'scheduler_options': cur_process_opts,
                            'name': self.get_hpcflow_job_name(task, 'process-task'),
                            'meta': {'from_tasks': [task.task_idx]},
                        }
                    ])
                else:
                    command_groups.append({
                        'directory': '.',
                        'nesting': 'hold',
                        'commands': task.get_process_task_commands(is_array=False),
                        'stats': False,
                        'scheduler_options': cur_process_opts,
                        'name': self.get_hpcflow_job_name(task, 'process-task'),
                        'meta': {'from_tasks': [task.task_idx]},
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

        if self.archives:

            command_groups[-1].update({
                'archive': self.archives[0],
                'archive_excludes': self.archive_excludes,
            })

            if len(self.archives) > 1:
                # Add a command group for each additional archive:
                for archive in self.archives[1:]:
                    command_groups.append({
                        'name': 'archive',
                        'directory': '.',
                        'commands': [
                            {
                                'line': f'echo "Archiving to {archive}!"'
                            },
                        ],
                        'archive': archive,
                        'archive_excludes': self.archive_excludes,
                    })

        if self.num_iterations > 1 or self.iterate:
            command_groups.append({
                'directory': '.',
                'nesting': 'hold',
                'commands': ('matflow write-element-directories '
                             '--iteration-idx=$(($ITER_IDX+1))'),
                'stats': False,
                'scheduler_options': self.iterate_run_options,
                'name': 'iterate',
            })

        hf_data = {
            'parallel_modes': Config.get('parallel_modes'),
            'scheduler': 'sge',
            'output_dir': 'output',
            'error_dir': 'output',
            'command_groups': command_groups,
            'variables': variables,
        }

        if self.num_iterations > 1:
            hf_data.update({
                'loop': {
                    'max_iterations': self.num_iterations,
                    'groups': list(range(len(command_groups))),
                }
            })

        elif self.iterate:

            # Find which command groups are to be repeated:
            iterate_groups = []
            for cmd_group_idx, cmd_group in enumerate(hf_data['command_groups']):
                if (
                    cmd_group['name'] == 'iterate' or
                    any([i in self.iterate['task_pathway']
                         for i in cmd_group['meta']['from_tasks']])
                ):
                    iterate_groups.append(cmd_group_idx)

            hf_data.update({
                'loop': {
                    'max_iterations': self.iterate['num_iterations'],
                    'groups': iterate_groups,
                }
            })

        for cmd_group_idx, cmd_group in enumerate(hf_data['command_groups']):
            # TODO: allow "meta" key in hpcflow command groups.
            hf_data['command_groups'][cmd_group_idx].pop('meta', None)

        if self.archives:
            hf_data.update({
                'archive_locations': {
                    archive: self.archive_definitions[archive]
                    for archive in self.archives
                }
            })

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

        # TODO: the following should be done by hpcflow?
        yaml_literal = scalarstring.LiteralScalarString
        for i in hf_data['command_groups']:

            if 'environment' in i and i['environment']:
                literal_str_env = yaml_literal(i['environment'].strip())
                i['environment'] = literal_str_env

            if isinstance(i['commands'], str):
                i['commands'] = yaml_literal(i['commands'])

            elif isinstance(i['commands'], list):
                for cmd in i['commands']:
                    if isinstance(cmd, dict) and 'subshell' in cmd:
                        if isinstance(cmd['subshell'], str):
                            cmd['subshell'] = yaml_literal(cmd['subshell'])

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

    def as_dict(self):
        """Return attributes dict with preceding underscores removed."""
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
        """Save the initial workflow to an HDF5 file and add task local inputs to the
        element data list.

        Parameters
        ----------
        path : str or Path, optional
            If specified, must be the full path (including file name) where the workflow
            file should be saved. By default, `None`, in which case the
            `default_file_path` attribute will be used as the full path.

        Notes
        -----
        The HDF5 file is written with two root-level groups:
            -  `element_data`: contains sub-groups which are hickle-loadable; each
                sub-group can be loaded as a dict that has the element parameter name
                and value.
            -   `workflow_obj`: a hickle-loadable dict-representation of the Workflow
                instance. Some sub-groups within `workflow_obj` are also hickle-loadable;
                these are: the `history` sub-group, and the parameter index dicts within
                each task element sub-group: `files_data_idx`, `inputs_data_idx` and
                `outputs_data_idx`. These give the `element_data` sub-group name (an
                integer) where that parameter data can be found.

        """

        # Generate base directory
        if self.path.exists():
            raise ValueError('Directories for this workflow already exist.')
        self.path.mkdir(exist_ok=False)

        path = Path(path or self.default_file_path)
        if path.exists():
            msg = f'Workflow cannot be saved to a path that already exists: "{path}".'
            raise WorkflowPersistenceError(msg)

        # Add local inputs to element_data list:
        element_data = {}
        for task in self.tasks:

            for input_name, vals_dict in task.local_inputs['inputs'].items():

                # Assumes no duplicate local inputs (distinguished by alias!):
                schema_input = task.schema.get_input_by_name(input_name)
                is_file = (schema_input['file'] != False)

                all_data_idx = []
                for val in vals_dict['vals']:
                    data_idx = len(element_data)
                    if is_file:
                        val = Path(val).name
                    element_data.update({(data_idx, input_name): val})
                    all_data_idx.append(data_idx)

                task.local_inputs['inputs'][input_name].update({
                    'vals_data_idx': [all_data_idx[i] for i in vals_dict['vals_idx']]
                })

        # Process imported data:
        for (param_name, param_context), imp_data in self.imported_parameters.items():
            for data_i in imp_data['data']:
                data_idx = len(element_data)
                element_data.update({(data_idx, param_name): data_i})
                imp_data['data_idx'].append(data_idx)
            del imp_data['data']

        workflow_as_dict = self.as_dict()
        obj_json = to_hicklable(workflow_as_dict)

        # Now dump workflow_obj and element_data individually:
        with h5py.File(path, 'w-') as handle:

            handle.attrs['matflow_version'] = __version__
            handle.attrs['workflow_id'] = self.id
            handle.attrs['workflow_version'] = 0

            workflow_group = handle.create_group('workflow_obj')
            hickle.dump(obj_json, handle, path=workflow_group.name)

            # Copy element parameter indices HDF5 attributes so they can still be loaded
            # as part of the whole workflow dict, after dumping individually:
            PARAM_IDX_NAMES = [
                'files_data_idx',
                'inputs_data_idx',
                'outputs_data_idx',
                'resource_usage',
            ]
            elem_param_idx_attrs = {}
            for task in self.tasks:
                elem_param_idx_attrs.update({task.task_idx: {}})
                for elem in task.elements:
                    elem_param_idx_attrs[task.task_idx].update({elem.element_idx: {}})
                    for i in PARAM_IDX_NAMES:
                        elem_param_idx_path = elem.HDF5_path + f"/'{i}'"
                        attrs = dict(handle[elem_param_idx_path].attrs)
                        elem_param_idx_attrs[task.task_idx][elem.element_idx][i] = attrs
                        del handle[elem_param_idx_path]

            # Dump element parameter indices individually:
            for task in self.tasks:
                for elem in task.elements:
                    elem_dict = elem.as_dict()
                    for i in PARAM_IDX_NAMES:
                        elem_param_idx_path = elem.HDF5_path + f"/'{i}'"
                        hickle.dump(
                            py_obj=elem_dict[i],
                            file_obj=handle,
                            path=elem_param_idx_path,
                        )
                        # Reinstate original hickle attributes for enabling full loading:
                        attrs = elem_param_idx_attrs[task.task_idx][elem.element_idx][i]
                        for k, v in attrs.items():
                            handle[elem_param_idx_path].attrs[k] = v

            # Dump history individually:
            hist_path = self.HDF5_path + "/'history'"
            hist_attrs = dict(handle[hist_path].attrs)
            del handle[hist_path]
            hickle.dump(
                py_obj=obj_json['history'],
                file_obj=handle,
                path=hist_path,
            )
            for k, v in hist_attrs.items():
                handle[hist_path].attrs[k] = v

            # Dump element data individually:
            data_group = handle.create_group('element_data')
            for (dat_idx, dat_name), dat_val in element_data.items():
                dat_key = Parameters.get_element_data_key(dat_idx, dat_name)
                hickle.dump(
                    py_obj=dat_val,
                    file_obj=handle,
                    path=data_group.name + '/' + dat_key
                )

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

        for i_idx, i in enumerate(obj_json['tasks']):
            i['status'] = TaskStatus(i['status'][1])
            for j in [
                'software_instance',
                'prepare_software_instance',
                'process_software_instance'
            ]:
                soft_inst_dict = i[j]
                machine = soft_inst_dict.pop('machine')
                soft_inst = SoftwareInstance(**soft_inst_dict)
                soft_inst.machine = machine
                i[j] = soft_inst
            for cmd_idx in range(len(i['schema']['command_group']['commands'])):
                del obj_json['tasks'][i_idx]['schema']['command_group']['commands'][cmd_idx]['options_raw']
                del obj_json['tasks'][i_idx]['schema']['command_group']['commands'][cmd_idx]['parameters_raw']
                del obj_json['tasks'][i_idx]['schema']['command_group']['commands'][cmd_idx]['stdin_raw']
                del obj_json['tasks'][i_idx]['schema']['command_group']['commands'][cmd_idx]['stdout_raw']
                del obj_json['tasks'][i_idx]['schema']['command_group']['commands'][cmd_idx]['stderr_raw']

        obj = {
            'name': obj_json['name'],
            'tasks': obj_json['tasks'],
            'stage_directory': obj_json['stage_directory'],
            'profile': obj_json['profile'],
            'extends': obj_json['extends'],
            'archives': obj_json['archives'],
            'archive_excludes': obj_json['archive_excludes'],
        }

        # For loading older workflow files without these attributes:
        WARN_ON_MISSING = [
            'figures',
            'metadata',
            'num_iterations',
            'iterate',
            'import_list',
            'imported_parameters',
        ]
        for key in WARN_ON_MISSING:
            if key not in obj_json:
                warn(f'"{key}" key missing from this workflow.')
            else:
                obj.update({key: obj_json[key]})

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

    def prepare_task_element(self, task_idx, element_idx, is_array=False):
        """
        Parameters
        ----------
        task_idx : int
        element_idx : int
        is_array : bool, optional
            If True, do not modify the workflow file directly, but save the new inputs
            and files to a separate HDF5 file within the element directory. After
            all task elements have been prepared, the results will be collated into the
            main workflow file.

        """

        if is_array:
            inputs_to_update = {}
            files_to_update = {}

        task = self.tasks[task_idx]
        element = task.elements[element_idx]

        # Populate element inputs:
        for input_alias, inputs_idx in task.elements_idx['inputs'].items():

            ins_task_idx = inputs_idx.get('task_idx')
            input_name = [i['name'] for i in task.schema.inputs
                          if i['alias'] == input_alias][0]

            if inputs_idx['element_idx'][element_idx] is not None:

                data_idx = []
                imp_key = inputs_idx['import_key'][element_idx]
                for src_elem_idx in inputs_idx['element_idx'][element_idx]:
                    if imp_key:
                        # Input values sourced from imported parameter:
                        imp_params = self.imported_parameters[imp_key]
                        param_data_idx = imp_params['data_idx'][src_elem_idx]
                    else:
                        # Input values sourced from previous task outputs:
                        src_all_elems = self.tasks[ins_task_idx[element_idx]].elements
                        src_element = src_all_elems[src_elem_idx]
                        param_data_idx = src_element.get_parameter_data_idx(input_name)
                    data_idx.append(param_data_idx)

                if inputs_idx['group'][element_idx] == 'default':
                    data_idx = data_idx[0]

            else:
                # Input values sourced from `local_inputs` of this task:
                local_data_idx = task.local_inputs['inputs'][input_name]['vals_data_idx']
                all_data_idx = [
                    (local_data_idx[i] if i is not None else None)
                    for i in inputs_idx['local_input_idx']
                ]
                data_idx = all_data_idx[element_idx]

            if is_array:
                inputs_to_update.update({input_alias: data_idx})
            else:
                element.add_input(input_alias, data_idx=data_idx)

        # Run input maps:
        schema_id = (task.name, task.method, task.software)
        in_map_lookup = Config.get('input_maps').get(schema_id)

        task_elem_path = self.get_element_path(task_idx, element_idx)

        # For each input file to be written, invoke the function:
        for in_map in task.schema.input_map:

            # Get inputs required for this file:
            in_map_inputs = {}
            for input_alias in in_map['inputs']:
                input_dict = task.schema.get_input_by_alias(input_alias)
                if input_dict.get('include_all_iterations'):
                    # Collate elements from all iterations. Need to get all elements at
                    # the same relative position within the iteration as this one:
                    all_iter_elems = self.get_elements_from_all_iterations(
                        task_idx,
                        element_idx,
                        up_to_current=True,
                    )
                    in_map_inputs.update({
                        input_alias: {
                            f'iteration_{iter_idx}': elem.get_input(input_alias)
                            for iter_idx, elem in enumerate(all_iter_elems)
                        }
                    })
                else:
                    in_map_inputs.update({input_alias: element.get_input(input_alias)})

            file_path = task_elem_path.joinpath(in_map['file'])

            # Run input map to generate required input files:
            func = in_map_lookup[in_map['file']]
            with working_directory(task_elem_path):
                func(path=file_path, **in_map_inputs)

            if in_map.get('save', False) and file_path.is_file():
                # Save generated file as string in workflow:
                with file_path.open('r') as handle:
                    file_dat = handle.read()
                if is_array:
                    files_to_update.update({in_map['file']: file_dat})
                else:
                    element.add_file(in_map['file'], value=file_dat)

        if is_array:
            temp_path = self._get_element_temp_array_prepare_path(task_idx, element_idx)
            dat = {'inputs': inputs_to_update, 'files': files_to_update}
            hickle.dump(dat, temp_path)

    @requires_path_exists
    def prepare_sources(self, task_idx, iteration_idx):
        """Prepare source files for the task preparation commands."""

        # Note: in future, we might want to parametrise the source function, which is
        # why we delay its invocation until task run time.

        if iteration_idx > 0:
            # Source files need to be generated only once per workflow (currently).
            return

        task = self.tasks[task_idx]

        if not task.software_instance.requires_sources:
            raise RuntimeError('The task has no sources to prepare.')

        source_map = Config.get('sources_maps')[(task.name, task.method, task.software)]
        source_func = source_map['func']
        source_files = source_func()

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
    def prepare_task(self, task_idx, iteration_idx, is_array=False):
        """Prepare inputs and run input maps.

        Parameters
        ----------
        task_idx : int
        is_array: bool, optional
            If True, prepare_task_element is assumed to have already run for each task,
            and we just need to collate the results from each element directory.

        """

        if (
            iteration_idx > 0 and
            self.iterate and
            task_idx not in self.iterate['task_pathway']
        ):
            # In the case where `prepare_task` for this task is in the same hpcflow
            # command group as a `process_task` from the previous task, which is
            # undergoing iteration.
            return

        task = self.tasks[task_idx]
        num_elems = self.elements_idx[task.task_idx]['num_elements_per_iteration']
        iter_elem_idx = [i + (iteration_idx * num_elems) for i in range(num_elems)]

        for element in index(task.elements, iter_elem_idx):

            if is_array:

                temp_path = self._get_element_temp_array_prepare_path(
                    task_idx,
                    element.element_idx,
                )
                dat = hickle.load(temp_path)
                inputs_to_update, files_to_update = dat['inputs'], dat['files']

                for input_alias, data_idx in inputs_to_update.items():
                    element.add_input(input_alias, data_idx=data_idx)

                for file_name, file_dat in files_to_update.items():
                    element.add_file(file_name, value=file_dat)

            else:
                self.prepare_task_element(
                    task.task_idx,
                    element.element_idx,
                    is_array=False,
                )

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
        """Execute a task that is to be run directly in Python (via the function mapper)."""

        task = self.tasks[task_idx]
        element = task.elements[element_idx]
        task_elem_path = self.get_element_path(task.task_idx, element_idx)
        schema_id = (task.name, task.method, task.software)
        func = Config.get('func_maps')[schema_id]

        all_inputs = {}
        for input_dict in task.schema.inputs:
            alias = input_dict['alias']
            if input_dict.get('include_all_iterations'):
                # Collate elements from all iterations. Need to get all elements at
                # the same relative position within the iteration as this one:
                all_iter_elems = self.get_elements_from_all_iterations(
                    task_idx,
                    element_idx,
                    up_to_current=True,
                )
                all_inputs.update({
                    alias: {
                        f'iteration_{iter_idx}': elem.get_input(alias)
                        for iter_idx, elem in enumerate(all_iter_elems)
                    }
                })
            else:
                all_inputs.update({alias: element.get_input(alias)})

        try:
            with working_directory(task_elem_path):
                outputs = func(**all_inputs) or {}
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

    def process_task_element(self, task_idx, element_idx, is_array=False):
        """
        Parameters
        ----------
        task_idx : int
        element_idx : int
        is_array : bool, optional
            If True, do not modify the workflow file directly, but save the new outputs
            and files to a separate HDF5 file within the element directory. After
            all task elements have been processed, the results will be collated into the
            main workflow file.

        """

        if is_array:
            outputs_to_update = {}
            files_to_update = {}

        task = self.tasks[task_idx]
        element = task.elements[element_idx]

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
        resource_usage = None
        for i in hf_sub_stats['command_group_submissions']:
            if i['name'] == job_name:
                resource_usage = i['tasks'][element_idx]
                if not is_array:
                    element.add_resource_usage(resource_usage)
                break

        if task.schema.is_func:
            func_outputs_path = self._get_element_temp_output_path(task_idx, element_idx)
            with func_outputs_path.open('rb') as handle:
                func_outputs = pickle.load(handle)
            for name, out in func_outputs.items():
                if is_array:
                    outputs_to_update.update({name: out})
                else:
                    element.add_output(name, value=out)

            func_outputs_path.unlink()

        task_elem_path = self.get_element_path(task.task_idx, element_idx)
        schema_id = (task.name, task.method, task.software)
        out_map_lookup = Config.get('output_maps').get(schema_id)

        # Run output maps:
        file_is_saved = []
        for out_map in task.schema.output_map:

            # Filter only those file paths required for this output:
            file_paths = []
            for i in out_map['files']:
                out_file_path = task_elem_path.joinpath(i['name'])
                file_paths.append(out_file_path)

                # Save generated file as string in workflow:
                if (
                    i['save'] and
                    out_file_path.is_file() and
                    i['name'] not in file_is_saved
                ):
                    file_is_saved.append(i['name'])
                    with out_file_path.open('r') as handle:
                        file_dat = handle.read()
                    if is_array:
                        files_to_update.update({i['name']: file_dat})
                    else:
                        element.add_file(i['name'], value=file_dat)

            func = out_map_lookup[out_map['output']]

            # Filter only output map options for this out_map:
            out_map_opts = {k: v for k, v in task.output_map_options.items()
                            if k in [i['name'] for i in out_map['options']]}

            # Get any input parameters to be passed to the out_map function:
            out_map_ins = {
                inp_dict['name']: element.get_input(inp_dict['name'])
                for inp_dict in out_map.get('inputs', [])
            }

            # Run output map:
            with working_directory(task_elem_path):
                try:
                    output = func(*file_paths, **out_map_ins, **out_map_opts)
                except Exception as err:
                    output = None
                    print(f'Failed to execute the output map for output '
                          f'"{out_map["output"]}". Exception was: {err}')

            if is_array:
                outputs_to_update.update({out_map['output']: output})
            else:
                element.add_output(out_map['output'], value=output)

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
                    file_dat = handle.read()
                if is_array:
                    outputs_to_update.update({output_name: file_dat})
                else:
                    element.add_output(output_name, value=file_dat)

        if is_array:
            temp_path = self._get_element_temp_array_process_path(task_idx, element_idx)
            dat = {
                'outputs': outputs_to_update,
                'files': files_to_update,
                'resource_usage': resource_usage,
            }
            hickle.dump(dat, temp_path)

        if task.cleanup:
            for file_glob in task.cleanup:
                for file_to_rm in task_elem_path.glob(file_glob):
                    if file_to_rm.is_file():
                        if task.id in str(file_to_rm):
                            # If the task ID is in the file name, then this is most-likely
                            # a temporary file generated by MatFlow and not to be deleted
                            # in this way!
                            print(f'Not deleting file (even though it matches): '
                                  f'{file_to_rm}')
                            continue
                        print(f'Deleting file: {file_to_rm}')
                        file_to_rm.unlink()

    @requires_path_exists
    def process_task(self, task_idx, iteration_idx, is_array=False):
        """Process outputs from an executed task: run output map and save outputs.

        Parameters
        ----------
        task_idx : int
        is_array: bool, optional
            If True, prepare_task_element is assumed to have already run for each task,
            and we just need to collate the results from each element directory.

        """

        if (
            iteration_idx > 0 and
            self.iterate and
            task_idx not in self.iterate['task_pathway']
        ):
            # In the case where `process_task` for this task is in the same hpcflow
            # command group as a `prepare_task` from the next task, which is undergoing
            # iteration.
            return

        task = self.tasks[task_idx]
        num_elems = self.elements_idx[task.task_idx]['num_elements_per_iteration']
        iter_elem_idx = [i + (iteration_idx * num_elems) for i in range(num_elems)]

        for element in index(task.elements, iter_elem_idx):

            if is_array:

                temp_path = self._get_element_temp_array_process_path(
                    task_idx,
                    element.element_idx,
                )
                dat = hickle.load(temp_path)

                for output_name, value in dat['outputs'].items():
                    element.add_output(output_name, value=value)

                for file_name, file_dat in dat['files'].items():
                    element.add_file(file_name, value=file_dat)

                element.add_resource_usage(dat['resource_usage'])

                temp_path.unlink()

            else:
                self.process_task_element(
                    task.task_idx,
                    element.element_idx,
                    is_array=False,
                )

        task.status = TaskStatus.complete
        self._append_history(WorkflowAction.process_task)

    def get_workflow_data(self, address):
        """Get workflow data according to its address"""
        output_name, address = address[0], address[1:]
        for task in self.tasks:
            for elem in task.elements:
                for output_param in list(elem.outputs.get_name_map().keys()):
                    if output_param == output_name:
                        output = elem.outputs.get(output_param)
                        out = get_nested_item(output, address)
                        return out

    def get_figure_data(self, fig_idx):
        """Get x-y data for a simple workflow figure from a figure spec"""
        fig_spec = self.figures[fig_idx]
        x_data = self.get_workflow_data(fig_spec['x'])
        y_data = self.get_workflow_data(fig_spec['y'])
        fig_data = {
            'x': x_data,
            'y': y_data,
        }
        return fig_data

    def get_figure_object(self, fig_idx, backend='plotly'):

        allowed_backs = ['plotly', 'matplotlib']
        if backend not in allowed_backs:
            raise ValueError(f'`backend` should be one of: {allowed_backs}')

        fig_spec = self.figures[fig_idx]
        fig_dat = self.get_figure_data(fig_idx)

        if backend == 'plotly':
            from plotly import graph_objects
            layout = {
                'xaxis_title': fig_spec.get('x_label'),
                'yaxis_title': fig_spec.get('y_label'),
                'margin_t': 35,
            }
            fig = graph_objects.Figure(fig_dat, layout)

        elif backend == 'matplotlib':
            from matplotlib import pyplot as plt
            plt.plot(fig_dat['x'], fig_dat['y'])
            fig = plt.gcf()
            plt.close()

        return fig

    def show_figure(self, fig_idx, backend='plotly'):
        fig = self.get_figure_object(fig_idx, backend)
        if backend == 'plotly':
            from plotly import graph_objects
            return graph_objects.FigureWidget(fig.data, fig.layout)
        elif backend == 'matplotlib':
            return fig

    @staticmethod
    def get_element_data_map(file_path):
        """Get the element_data group name for all inputs/outputs/files.

        Parameters
        ----------
        file_path : str or Path
            Path to the HDF5 workflow file.

        Returns
        -------
        data_map : list of list of dict
            For each task and for each element, a dict for inputs, outputs and files is
            returned that maps the parameter to its /element_data group name.
        """

        with h5py.File(str(file_path), 'r') as handle:

            idx_map = {int(re.search('(\d+)', i).group()): i
                       for i in handle['/element_data']}

            tasks_path = "/workflow_obj/data/'tasks'/data"
            task_lists = []
            for task_group in handle[tasks_path].values():
                element_path = task_group.name + "/'elements'/data"

                element_dicts = []
                for elem_group in handle[element_path].values():
                    params_paths = {
                        'inputs': elem_group.name + "/'inputs_data_idx'/data",
                        'outputs': elem_group.name + "/'outputs_data_idx'/data",
                        'files': elem_group.name + "/'files_data_idx'/data",
                    }
                    params_dict = {k: {} for k in params_paths.keys()}
                    for param_type, param_path in params_paths.items():
                        for param_name_quoted, param_group in handle[param_path].items():
                            param_name = param_name_quoted[1:-1]
                            elem_idx = param_group['data'][()]
                            if isinstance(elem_idx, np.ndarray):
                                elem_idx = elem_idx.tolist()
                            else:
                                elem_idx = [elem_idx]
                            params_dict[param_type].update(
                                {param_name: [idx_map[i] for i in elem_idx]}
                            )
                    element_dicts.append(params_dict)

                task_lists.append(element_dicts)

        return task_lists

    @staticmethod
    def get_all_element_parameters(file_path, task_idx, element_idx, convert_numpy=False):
        data_map = Workflow.get_element_data_map(file_path)
        all_params = {
            'inputs': {},
            'outputs': {},
            'files': {},
        }
        with h5py.File(file_path, 'r') as handle:
            for param_type, params in data_map[task_idx][element_idx].items():
                for name, data_idx in params.items():
                    all_params[param_type].update({name: []})
                    for data_idx_i in data_idx:
                        dat_path = f'/element_data/{data_idx_i}'
                        dat = hickle.load(handle, path=dat_path)
                        if convert_numpy:
                            dat = nested_dict_arrays_to_list(dat)
                        all_params[param_type][name].append(dat)

        return all_params

    @staticmethod
    def get_task_parameter_data(file_path, task_idx, convert_numpy=False):
        data_map = Workflow.get_element_data_map(file_path)
        all_elems = []
        with h5py.File(file_path, 'r') as handle:
            for elem in data_map[task_idx]:
                all_params = {
                    'inputs': {},
                    'outputs': {},
                    'files': {},
                }
                for param_type, params in elem.items():
                    for name, data_idx in params.items():
                        all_params[param_type].update({name: []})
                        for data_idx_i in data_idx:
                            dat_path = f'/element_data/{data_idx_i}'
                            dat = hickle.load(handle, path=dat_path)
                            if convert_numpy:
                                dat = nested_dict_arrays_to_list(dat)
                            all_params[param_type][name].append(dat)
                all_elems.append(all_params)

        return all_elems

    @staticmethod
    def swap_task_parameter_data_indexing(task_parameter_data):
        """Restructure the task parameter data such that the multiple
        element values of a given parameter are located together in a list
        for each parameter."""
        out = {
            'inputs': {},
            'outputs': {},
            'files': {},
        }
        for elem_dat in task_parameter_data:
            for param_type in out.keys():
                for param_name, param_val in elem_dat[param_type].items():
                    if param_name not in out[param_type]:
                        out[param_type].update({param_name: [param_val]})
                    else:
                        out[param_type][param_name].append(param_val)
        return out

    @staticmethod
    def get_schema_info(file_path, task_idx):
        """Get schema input aliases, output names, input and output maps directly
        from the HDF5 file."""

        task_group_path = f"/workflow_obj/data/'tasks'/data/data_{task_idx}"
        schema_path = task_group_path + f"/'schema'/data"

        with h5py.File(file_path, 'r') as handle:

            input_aliases = []
            ins_path = schema_path + f"/'inputs'/data"
            for ins_group in handle[ins_path].values():
                ins_alias_path = ins_group.name + f"/'alias'/data"
                ins_alias = handle[ins_alias_path][()]
                input_aliases.append(ins_alias)

            outputs = [i.decode() for i in handle[schema_path + f"/'outputs'/data"][()]]

            input_maps = []
            ins_maps_path = schema_path + f"/'input_map'/data"
            for in_map_group in handle[ins_maps_path].values():
                in_map_file_path = in_map_group.name + f"/'file'/data"
                in_map_file = handle[in_map_file_path][()]
                in_map_ins = [i.decode()
                              for i in handle[in_map_group.name + f"/'inputs'/data"][()]]
                in_map_i = {
                    'file': in_map_file,
                    'inputs': in_map_ins,
                }
                input_maps.append(in_map_i)

            output_maps = []
            outs_maps_path = schema_path + f"/'output_map'/data"
            for out_map_group in handle[outs_maps_path].values():
                out_map_output_path = out_map_group.name + f"/'output'/data"
                out_map_output = handle[out_map_output_path][()]
                out_map_files = []
                for i in handle[out_map_group.name + f"/'files'/data"].values():
                    omf_name_path = i.name + f"/'name'/data"
                    out_map_files.append(handle[omf_name_path][()])
                out_map_i = {
                    'files': out_map_files,
                    'output': out_map_output,
                }
                output_maps.append(out_map_i)

        schema_info = {
            'input_aliases': input_aliases,
            'outputs': outputs,
            'input_map': input_maps,
            'output_map': output_maps,
        }
        return schema_info

    @staticmethod
    def get_task_name_friendly(file_path, task_idx):
        task_name_path = f"/workflow_obj/data/'tasks'/data/data_{task_idx}/'name'/data"
        with h5py.File(file_path, 'r') as handle:
            task_name = handle[task_name_path][()]
            name = '{}{}'.format(task_name[0].upper(), task_name[1:]).replace('_', ' ')
            return name

    @staticmethod
    def get_workflow_tasks_info(file_path):
        tasks_info = []
        tasks_path = "/workflow_obj/data/'tasks'/data"
        with h5py.File(file_path, 'r') as handle:
            for task_idx, task_group in enumerate(handle[tasks_path].values()):
                name = handle[task_group.name + "/'name'/data"][()]
                name_friendly = '{}{}'.format(name[0].upper(), name[1:]).replace('_', ' ')
                schema_method = handle[task_group.name +
                                       "/'schema'/data/'method'/data"][()]
                schema_impl = handle[task_group.name +
                                     "/'schema'/data/'implementation'/data"][()]
                task_dict = {
                    'name': name,
                    'name_friendly': name_friendly,
                    'task_idx': task_idx,
                    'schema_method': schema_method,
                    'schema_implementation': schema_impl,
                }
                tasks_info.append(task_dict)
        return tasks_info

    def get_input_tasks(self, parameter_name, context=None):
        """Return task indices of tasks in which a given parameter is an input.

        Parameters
        ----------
        parameter_name : str
            Name of the parameter to locate.
        context : str, optional
            If specified, limit the search to only tasks with the given context.

        Returns
        -------
        input_task_idx : list of int
            List of task indices in which the specified parameter features as an input.

        """

        input_task_idx = []
        for task in self.tasks:
            if parameter_name in task.schema.input_names:
                if (context is not None and task.context == context) or context is None:
                    input_task_idx.append(task.task_idx)

        return input_task_idx

    def get_output_tasks(self, parameter_name, context=None):
        """Return task indices of tasks in which a given parameter is an output.

        Parameters
        ----------
        parameter_name : str
            Name of the parameter to locate.
        context : str, optional
            If specified, limit the search to only tasks with the given context.

        Returns
        -------
        output_task_idx : list of int
            List of task indices in which the specified parameter features as an output.

        """

        output_task_idx = []
        for task in self.tasks:
            if parameter_name in task.schema.outputs:
                if (context is not None and task.context == context) or context is None:
                    output_task_idx.append(task.task_idx)

        return output_task_idx

    def get_dependent_tasks(self, task_idx, recurse=False):
        """Get the indices of tasks that depend on a given task.

        Notes
        -----
        For the inverse, see `get_task_dependencies`.

        """

        out = []
        for idx, dep_idx in enumerate(self.dependency_idx):
            if task_idx in dep_idx['task_dependencies']:
                out.append(idx)

        if recurse:
            out += list(set([
                additional_out
                for task_idx_i in out
                for additional_out in
                self.get_dependent_tasks(task_idx_i, recurse=True)
            ]))

        return out

    def get_task_dependencies(self, task_idx, recurse=False):
        """Get the indicies of tasks that a given task depends on.

        Notes
        -----
        For the inverse, see `get_dependent_tasks`.

        """

        out = self.dependency_idx[task_idx]['task_dependencies']

        if recurse:
            out += list(set([
                additional_out
                for task_idx_i in out
                for additional_out in
                self.get_task_dependencies(task_idx_i, recurse=True)
            ]))

        return out

    def get_dependent_parameters(self, parameter_name, recurse=False, return_list=False):
        """Get the names of parameters that depend on a given parameter.

        Parameters
        ----------
        parameter_name : str
        recurse : bool, optional
            If False, only include output parameters from tasks for which the given
            parameter is an input. If True, include output parameters from dependent tasks
            as well. By default, False.
        return_list : bool, optional
            If True, return a list of output parameters. If False, return a dict whose
            keys are the task indices and whose values are the output parameters from each
            task. By default, False.

        Returns
        -------
        dict of (int : list of str) or list of str        

        Notes
        -----
        For the inverse, see `get_parameter_dependencies`.

        """

        # Get the tasks where given parameter is an input:
        all_task_idx = self.get_input_tasks(parameter_name)

        # If recurse, need outputs from dependent tasks as well:
        if recurse:
            all_task_idx += list(set([
                additional_task
                for task_idx_i in all_task_idx
                for additional_task in
                self.get_dependent_tasks(task_idx_i, recurse=True)
            ]))

        # Get output parameters from tasks:
        params = {
            task_idx: self.tasks[task_idx].schema.outputs
            for task_idx in all_task_idx
        }

        if return_list:
            params = list(set([i for param_vals in params.values() for i in param_vals]))

        return params

    def get_parameter_dependencies(self, parameter_name, recurse=False, return_list=False):
        """Get the names of parameters that a given parameter depends on.

        Parameters
        ----------
        parameter_name : str
        recurse : bool, optional
            If False, only include input parameters from tasks for which the given
            parameter is an output. If True, include input parameters from task
            dependencies as well. By default, False.
        return_list : bool, optional
            If True, return a list of input parameters. If False, return a dict whose keys
            are the task indices and whose values are the input parameters from each task.
            By default, False.

        Returns
        -------
        dict of (int : list of str) or list of str

        Notes
        -----
        For the inverse, see `get_dependent_parameters`.

        """

        # Get the tasks where given parameter is an output
        all_task_idx = self.get_output_tasks(parameter_name)

        # If recurse, need inputs from tasks dependencies as well:
        if recurse:
            all_task_idx = list(set([
                additional_task
                for task_idx_i in all_task_idx
                for additional_task in
                self.get_task_dependencies(task_idx_i, recurse=True)
            ])) + all_task_idx

        # Get input parameters from tasks:
        params = {
            task_idx: self.tasks[task_idx].schema.input_names
            for task_idx in all_task_idx
        }

        if return_list:
            params = list(set([i for param_vals in params.values() for i in param_vals]))

        return params

    def get_iteration_task_pathway(self, parameter_name):

        originating_tasks = self.get_input_tasks(parameter_name)
        dep_tasks = [j for i in originating_tasks
                     for j in self.get_dependent_tasks(i, recurse=True)]

        # Which dep_tasks produces the iteration parameter?
        outputs_iter_param = [parameter_name in self.tasks[i].schema.outputs
                              for i in dep_tasks]

        if not any(outputs_iter_param):
            msg = (f'Parameter "{parameter_name}" is not output by any task and so '
                   f'cannot be iterated.')
            raise WorkflowIterationError(msg)

        # Consider the task with the largest task_idx that outputs the iteration parameter:
        # (in the case of multiple parameter-modifying tasks):
        producing_task_trials = [i for idx, i in enumerate(dep_tasks)
                                 if outputs_iter_param[idx]]
        producing_task_idx = max(producing_task_trials)

        task_pathway = list(set(originating_tasks + dep_tasks))
        out = {
            'task_pathway': task_pathway,
            'originating_tasks': originating_tasks,
            'producing_task': producing_task_idx,
        }

        return out

    def get_elements_from_all_iterations(self, task_idx, element_idx, up_to_current=True):
        """
        Get equivalent elements from all iterations.

        Parameters
        ----------
        task_idx : int
        element_idx : int
        up_to_current : bool, optional
            If True, only return elements from iterations up to and including the
            iteration of the given element. If False, return elements from all iterations.

        Returns
        -------
        list of Element

        """

        elems_idx_i = self.elements_idx[task_idx]

        iter_idx_bool = np.zeros_like(elems_idx_i['iteration_idx'], dtype=bool)
        iter_idx_bool[element_idx] = True
        iter_idx_reshape = np.array(iter_idx_bool).reshape(
            (elems_idx_i['num_iterations'],
             elems_idx_i['num_elements_per_iteration'])
        )

        current_iter, idx_within_iteration = [i[0] for i in np.where(iter_idx_reshape)]
        if not up_to_current:
            iter_idx_reshape[:, idx_within_iteration] = True
        else:
            for i in range(current_iter):
                iter_idx_reshape[i, idx_within_iteration] = True

        all_elem_idx = np.where(iter_idx_reshape.flatten())[0]
        ell_elems = [self.tasks[task_idx].elements[i] for i in all_elem_idx]

        return ell_elems

    def do_archive(self, archive):
        """Perform an on-demand archive outside of the normal workflow submission process.

        Parameters
        ----------
        archive : str
            Name of the archive. Must exist in the MatFlow config YAML file.

        """
        command_groups = [
            {
                'name': 'archive',
                'directory': '.',
                'commands': [
                    {
                        'line': 'echo "Archiving!"'
                    },
                ],
                'archive': archive,
                'archive_excludes': self.archive_excludes,
            },
        ]
        archive_defn = {
            **Config.get('archive_locations')[archive],
            'root_directory_name': 'parent',
        }
        hf_data = {
            'scheduler': 'sge',
            'output_dir': 'output',
            'error_dir': 'output',
            'command_groups': command_groups,
            'archive_locations': {archive: archive_defn},
        }
        hf_wid = hpcflow.make_workflow(
            dir_path=self.path,
            workflow_dict=hf_data,
            config_dir=Config.get('hpcflow_config_dir'),
        )
        hpcflow.submit_workflow(
            workflow_id=hf_wid,
            dir_path=self.path,
            config_dir=Config.get('hpcflow_config_dir'),
        )
