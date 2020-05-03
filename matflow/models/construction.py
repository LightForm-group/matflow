"""matflow.models.construction.py

Functions for constructing model (Workflow/Task) attributes. This functionality is mostly
contained here, rather than within the Workflow/Task classes, because Workflow/Task
objects can either be constructed from scratch, or reconstituted from the persistent
workflow file. This way initialisation of these objects is always consistent, and the
classes themselves mainly serve to provide useful properties.

"""

import copy
from warnings import warn

import numpy as np

from matflow import SOFTWARE, TASK_SCHEMAS
from matflow.errors import (
    IncompatibleWorkflow,
    MissingSoftware,
    WorkflowPersistenceError,
    SequenceError,
    IncompatibleSequence,
    TaskError,
)
from matflow.utils import tile, repeat, arange
from matflow.models.task import Task, TaskSchema


def get_schema_dict(name, method, software_instance=None):
    """Get the schema associated with the method/implementation of this task."""

    match_task_idx = None
    match_method_idx = None
    match_imp_idx = None

    for task_ref_idx, task_ref in enumerate(TASK_SCHEMAS):

        if task_ref['name'] == name:

            match_task_idx = task_ref_idx
            for met_idx, met in enumerate(task_ref['methods']):

                if met['name'] == method:

                    match_method_idx = met_idx
                    implementations = met.get('implementations')
                    if implementations:

                        for imp_idx, imp in enumerate(implementations):

                            if imp['name'] == software_instance['name']:
                                match_imp_idx = imp_idx
                                break
                    break
            break

    if match_task_idx is None:
        msg = (f'No matching task found with name: "{name}"')
        raise ValueError(msg)

    if match_method_idx is None:
        msg = (f'No matching method found with name: "{method}"'
               f' in task: "{name}""')
        raise ValueError(msg)

    task_ref = TASK_SCHEMAS[match_task_idx]
    met_ref = task_ref['methods'][met_idx]
    inputs = task_ref.get('inputs', []) + met_ref.get('inputs', [])
    outputs = task_ref.get('outputs', []) + met_ref.get('outputs', [])

    imp_ref = None
    in_map = None
    out_map = None
    command_opt = None

    if match_imp_idx is not None:
        imp_ref = met_ref['implementations'][match_imp_idx]

        inputs += imp_ref.get('inputs', [])
        outputs += imp_ref.get('outputs', [])

        in_map = imp_ref.get('input_map', [])
        out_map = imp_ref.get('output_map', [])
        command_opt = imp_ref.get('commands', [])

    outputs = list(set(outputs))

    if software_instance:
        implementation = software_instance['name']
        command_group = {
            'commands': command_opt,
            'env_pre': software_instance.get('env_pre'),
            'env_post': software_instance.get('env_post'),
        }
    else:
        implementation = None
        command_group = None

    schema_dict = {
        'name': name,
        'method': method,
        'implementation': implementation,
        'inputs': inputs,
        'outputs': outputs,
        'input_map': in_map,
        'output_map': out_map,
        'command_group': command_group,
    }

    return schema_dict


def normalise_local_inputs(base=None, sequences=None):
    'Validate and normalise sequences and task inputs for a given task.'

    if base is None:
        base = {}
    if sequences is None:
        sequences = []

    if not isinstance(sequences, list):
        raise SequenceError('`sequences` must be a list.')

    nest_req = True if len(sequences) > 1 else False

    req_seq_keys = ['name', 'vals']
    allowed_seq_keys = req_seq_keys + ['nest_idx']

    prev_num_vals = None
    prev_nest = None
    inputs_lst = []
    for seq in sequences:

        miss_keys = list(set(req_seq_keys) - set(seq.keys()))
        if miss_keys:
            miss_keys_fmt = ', '.join([f'"{i}"' for i in miss_keys])
            msg = f'Missing keys from sequence definition: {miss_keys_fmt}.'
            raise SequenceError(msg)

        bad_keys = list(set(seq.keys()) - set(allowed_seq_keys))
        if bad_keys:
            bad_keys_fmt = ', '.join([f'"{i}"' for i in bad_keys])
            raise SequenceError(f'Unknown keys from sequence definition: {bad_keys_fmt}.')

        if not isinstance(seq['vals'], list):
            raise SequenceError(f'Sequence "{seq["name"]}" `vals` must be a list.')

        if 'nest_idx' in seq:
            if not isinstance(seq['nest_idx'], int) or (seq['nest_idx'] < 0):
                msg = (f'`nest_idx` must be a positive integer or zero for sequence '
                       f'"{seq["name"]}"')
                raise SequenceError(msg)

        if nest_req:
            if 'nest_idx' not in seq:
                msg = f'`nest_idx` is required for sequence "{seq["name"]}".'
                raise SequenceError(msg)
        else:
            if 'nest_idx' in seq:
                warn(f'`nest_idx` is not required for sequence "{seq["name"]}.')
            seq['nest_idx'] = 0  # set a default

        nest = seq['nest_idx']

        num_vals = len(seq['vals'])
        if prev_num_vals and nest == prev_nest:
            # For same nest_idx, sequences must have the same lengths:
            if num_vals != prev_num_vals:
                msg = (f'Sequence "{seq["name"]}" shares a `nest_idx` with another '
                       f'sequence but has a different number of values.')
                raise IncompatibleSequence(msg)

        prev_num_vals = num_vals
        prev_nest = nest
        inputs_lst.append(copy.deepcopy(seq))

    for in_name, in_val in base.items():
        inputs_lst.append({
            'name': in_name,
            'nest_idx': -1,
            'vals': [copy.deepcopy(in_val)],
        })

    inputs_lst.sort(key=lambda x: x['nest_idx'])

    return inputs_lst


def get_local_inputs(schema_inputs, base=None, num_repeats=1, sequences=None,
                     nest=True, merge_priority=None, groups=None):
    """Combine task base/sequences/repeats to get the locally defined inputs for a task.

    Parameters
    ----------
    schema_inputs: list of str

    """

    inputs_lst = normalise_local_inputs(base, sequences)

    if inputs_lst:
        lengths = [len(i['vals']) for i in inputs_lst]
        total_len = len(inputs_lst[0]['vals'])
        for idx, i in enumerate(inputs_lst[1:], 1):
            if i['nest_idx'] > inputs_lst[idx - 1]['nest_idx']:
                total_len *= len(i['vals'])

        prev_reps = total_len
        prev_tile = 1
        prev_nest = None

    else:
        total_len = 1

    local_ins = {'inputs': {}}

    for idx, input_i in enumerate(inputs_lst):

        if (prev_nest is None) or (input_i['nest_idx'] > prev_nest):
            rep_i = int(prev_reps / lengths[idx])
            tile_i = int(total_len / (lengths[idx] * rep_i))
            prev_nest = input_i['nest_idx']
        else:
            rep_i = prev_reps
            tile_i = prev_tile

        vals_idx = tile(repeat(arange(lengths[idx]), rep_i), tile_i)
        vals_idx = repeat(vals_idx, num_repeats)
        repeats_idx = tile(arange(num_repeats), total_len)

        local_ins['repeats_idx'] = repeats_idx
        local_ins['inputs'].update({
            input_i['name']: {
                # 'nest_idx': input_i['nest_idx'],
                'vals': input_i['vals'],
                'vals_idx': vals_idx,
            }
        })
        prev_reps = rep_i
        prev_tile = tile_i

    allowed_grp = schema_inputs + ['repeats']
    allowed_grp_fmt = ', '.join([f'"{i}"' for i in allowed_grp])

    def_group = {'default': {'nest': nest, 'group_by': allowed_grp}}
    if merge_priority:
        def_group.update({'merge_priority': merge_priority})

    user_groups = {}
    for group_name, group in (groups or {}).items():

        if 'group_by' not in group:
            raise ValueError(f'Missing `group_by` key in group {group_name}.')

        for param in group['group_by']:
            if param not in allowed_grp:
                msg = (f'Parameter "{param}" cannot be grouped, because it '
                       f'has no specified values. Allowed group values are: '
                       f'{allowed_grp_fmt}.')
                raise ValueError(msg)

        user_groups.update({f'user_group_{group_name}': group})

    all_groups = {**def_group, **user_groups}

    local_ins['groups'] = all_groups
    local_ins['length'] = total_len

    return local_ins


def get_software_instance(software_name, num_cores):
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


def get_dependency_idx(task_info_lst):
    """Find the dependencies between tasks.

    Parameters
    ----------
    task_info_lst : list of dict
        Each dict must have keys:
            context : str
            schema : TaskSchema

    Returns
    -------
    dependency_idx : list of list of int
        Each element, which corresponds to a given task in `task_info_list`, 
        lists the task indices upon which the given task depends.

    Notes
    -----
    - Two conditions must be met for a task (the downstream task) to be recorded
      as depending on another (upstream) task: 
          1) one of the downstream task's input parameters must be one of the
             upstream task's output parameters;
          2) EITHER:
              - One of the downstream task's input parameters shares a context
                with the upstream task, OR
              - The upstream and downstream task share the same context, and,
                for any downstream task input parameter, the parameter context
                is `None`.             

    """

    dependency_idx = []
    all_outputs = []
    for task_info in task_info_lst:

        downstream_context = task_info['context']
        schema_inputs = task_info['schema'].inputs
        schema_outputs = task_info['schema'].outputs

        # List outputs with their corresponding task contexts:
        all_outputs.extend([(i, downstream_context) for i in schema_outputs])

        # Find which tasks this task depends on:
        output_idx = []
        for input_j in schema_inputs:

            param_name = input_j['name']
            param_context = input_j['context']

            for task_idx_k, task_info_k in enumerate(task_info_lst):

                if param_name not in task_info_k['schema'].outputs:
                    continue

                upstream_context = task_info_k['context']
                if (
                    param_context == upstream_context or (
                        (upstream_context == downstream_context) and
                        (param_context is None)
                    )
                ):
                    output_idx.append(task_idx_k)

        dependency_idx.append(list(set(output_idx)))

    if len(all_outputs) != len(set(all_outputs)):
        msg = 'Multiple tasks in the workflow have the same output and context!'
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


def validate_task_dict(task, is_from_file, check_integrity=True):
    """Validate a task dict.

    Parameters
    ----------
    task : dict
        Keys are either those specified for the task when generating a new
        workflow, or the full set of task attributes loaded from a task within
        a workflow file.
    is_from_file : bool
        Has this task dict been loaded from a workflow file or is it associated
        with a brand new workflow?
    check_integrity : bool, optional
        Applicable if `is_from_file` is True. If True, re-generate `local_inputs`
        and compare them to those loaded from the file. If the equality test
        fails, raise. True by default. If False, `local_inputs` are still
        re-generated, but they are not compared to the loaded `local_inputs`.

    Returns
    -------
    task : dict
        Copy of original `task` dict, with default keys added. If `is_from_file=False`,
        `local_inputs` and `schema` are added.

    """

    if not isinstance(task, dict):
        raise TaskError(f'Task definition must be a dict, but "{type(task)}" given.')

    if is_from_file:
        req_keys = [
            'name',
            'method',
            'software_instance',
            'task_idx',
            'run_options',
            'status',
            'stats',
            'context',
            'local_inputs',
            'inputs',
            'outputs',
            'schema',
            'files',
            'resource_usage',
            'base',
            'sequences',
            'repeats',
            'groups',
            'nest',
            'merge_priority',
        ]
        good_keys = req_keys
        def_keys = {}
    else:
        req_keys = ['name', 'software', 'method']
        good_keys = [
            'software',
            'run_options',
            'stats',
            'context',
            'base',
            'sequences',
            'repeats',
            'groups',
            'nest',
            'merge_priority',
        ] + req_keys

        def_keys = {
            'run_options': {'num_cores': 1},
            'stats': True,
            'context': '',
            'base': None,
            'sequences': None,
            'repeats': 1,
            'groups': None,
            'nest': True,
            'merge_priority': None,
        }

    miss_keys = list(set(req_keys) - set(task.keys()))
    if miss_keys:
        miss_keys_fmt = ', '.join([f'"{i}"' for i in miss_keys])
        msg = (f'Missing keys in Task definition (`is_from_file={is_from_file}`): '
               f'{miss_keys_fmt}.')
        raise TaskError(msg)

    bad_keys = list(set(task.keys()) - set(good_keys))
    if bad_keys:
        bad_keys_fmt = ', '.join([f'"{i}"' for i in bad_keys])
        msg = (f'Unknown keys in Task definition (`is_from_file={is_from_file}`): '
               f'{bad_keys_fmt}.')
        raise TaskError(msg)

    task = {**def_keys, **copy.deepcopy(task)}

    # Make TaskSchema:
    if is_from_file:
        # Load from file (don't rely on the task schema existing on this installation):
        schema = TaskSchema(**task['schema'])

    else:
        # Find the software instance:
        soft_inst = get_software_instance(
            task['software'],
            task['run_options']['num_cores'],
        )
        schema_dict = get_schema_dict(task['name'], task['method'], soft_inst)
        schema = TaskSchema(**schema_dict)

    local_ins = get_local_inputs(
        schema.input_names,
        base=task['base'],
        num_repeats=task['repeats'],
        sequences=task['sequences'],
        nest=task['nest'],
        merge_priority=task['merge_priority'],
        groups=task['groups'],
    )

    if is_from_file and check_integrity:
        if local_ins != task['local_inputs']:
            msg = (
                f'Regenerated local inputs (task: "{task["name"]}") '
                f'are not equivalent to those loaded from the '
                f'workflow file. Stored local inputs are:'
                f'\n{task["local_inputs"]}\nRegenerated local '
                f'inputs are:\n{local_ins}\n.'
            )
            raise WorkflowPersistenceError(msg)

    task['local_inputs'] = local_ins
    task['schema'] = schema

    return task


def check_consistent_inputs(task_lst, dep_idx):
    """Check for missing and surplus inputs for each in a list of task dicts.

    Parameters
    ----------
    task_lst : list of dict
        Each dict must have keys:
            schema : TaskSchema
                Task schema against which to validate the inputs.
            local_input_names : list of str
                List of the locally defined inputs for the task.
    dep_idx : list of list of int
        List of length equal to original `task_lst`, whose elements are integer
        lists that link a given task to the indices of tasks upon which it
        depends.    

    """

    for dep_idx_i, task in zip(dep_idx, task_lst):

        defined_inputs = task['local_input_names']
        task['schema'].check_surplus_inputs(defined_inputs)

        task_inp_names = [k['name'] for k in task['schema'].inputs]
        for j in dep_idx_i:
            for output in task_lst[j]['schema'].outputs:
                if output in task_inp_names:
                    defined_inputs.append(output)

        task['schema'].check_missing_inputs(defined_inputs)


def order_tasks(task_lst):
    """Order tasks according to their dependencies.

    Parameters
    ----------
    task_lst : list of dict
        List of dicts representing the validated task attributes. Each dict
        must contain the keys:
            context : str
            schema : TaskSchema

    Returns
    -------
    task_lst_srt : list of dict
        Task list ordered to satisfy the dependencies between tasks, such that
        a task that depends on another task is placed after that other task.
    dep_idx_srt : list of list of int
        List of length equal to original `task_lst`, whose elements are integer
        lists that link a given task to the indices of tasks upon which it
        depends.

    """

    dep_idx = get_dependency_idx(task_lst)

    task_lst_check = [
        {
            'schema': i['schema'],
            'local_input_names': i['local_inputs']['inputs'].keys(),
        } for i in task_lst
    ]
    check_consistent_inputs(task_lst_check, dep_idx)

    # Find the index at which each task must be positioned to satisfy input
    # dependencies, and reorder tasks (and `dep_idx`!):
    min_idx = [max(i or [0]) + 1 for i in dep_idx]
    task_srt_idx = np.argsort(min_idx)

    task_lst_srt = [task_lst[i] for i in task_srt_idx]
    dep_idx_srt = [[np.argsort(task_srt_idx)[j] for j in dep_idx[i]]
                   for i in task_srt_idx]

    # Add sorted task idx:
    for idx, i in enumerate(task_lst_srt):
        i['task_idx'] = idx

    return task_lst_srt, dep_idx_srt


def init_tasks(task_lst, is_from_file, check_integrity=True):
    """Construct and validate Task objects and the element indices
    from which to populate task inputs.

    Parameters
    ----------
    task_lst : list of dict
        List of task definitions. Each task dict can have the following
        keys:
            name
    is_from_file : bool
        If True, assume we are loading tasks from an HDF5 file, otherwise,
        the tasks are being constructed from an entirely new Workflow.
    check_integrity : bool, optional
        Applicable if `is_from_file=True`. It True, do some checks that the
        loaded information makes sense. True by default.

    Returns
    -------
    tasks : list of Task
    element_idx : list of dict

    Notes
    -----
    If loading an existing workflow, there will be additional keys in the
    dict elements of `task_lst`.

    """

    # Validate and add `schema` and `local_inputs` to each task:
    task_lst = [validate_task_dict(i, is_from_file, check_integrity) for i in task_lst]

    # Get dependencies, sort and add `task_idx` to each task:
    task_lst, dep_idx = order_tasks(task_lst)

    # Find element indices that determine the elements from which task inputs are drawn:
    element_idx = []

    # Create list of Task objects:
    task_objs = [Task(**i) for i in task_lst]

    return task_objs, element_idx
