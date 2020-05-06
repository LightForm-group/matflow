"""matflow.models.construction.py

This is the business end of matflow.

Functions for constructing model (Workflow/Task) attributes. This functionality is mostly
contained here, rather than within the Workflow/Task classes, because Workflow/Task
objects can either be constructed from scratch, or reconstituted from the persistent
workflow file. This way initialisation of these objects is always consistent, and testing
is easier. The classes themselves mainly serve to provide useful properties.

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
    MissingMergePriority,
    IncompatibleTaskNesting,
    UnsatisfiedGroupParameter,
)
from matflow.utils import (tile, repeat, arange, extend_index_list, flatten_list,
                           to_sub_list)
from matflow.models.task import Task, TaskSchema


def get_schema_dict(name, method, all_task_schemas, software_instance=None):
    """Get the schema associated with the method/implementation of this task."""

    match_task_idx = None
    match_method_idx = None
    match_imp_idx = None

    for task_ref_idx, task_ref in enumerate(all_task_schemas):

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

    task_ref = all_task_schemas[match_task_idx]
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
    local_ins['length'] = total_len * num_repeats

    return local_ins


def get_software_instance(software_name, num_cores, all_software):
    """Find a software instance in the software.yml file that matches the software
    requirements of a given task."""

    for soft_inst in all_software:

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


def validate_task_dict(task, is_from_file, all_software, all_task_schemas,
                       check_integrity=True):
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
    all_software
    all_task_schemas
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
            'run_options': {},
            'stats': True,
            'context': '',
            'base': None,
            'sequences': None,
            'repeats': 1,
            'groups': None,
            'merge_priority': None,
            'nest': True,
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
    if 'num_cores' not in task['run_options']:
        task['run_options'].update({'num_cores': 1})

    # (SGE specific):
    if task['run_options']['num_cores'] > 1:
        if 'pe' not in task['run_options']:
            msg = ('Parallel environment (`pe`) key must be specified in `run_options`, '
                   'since `num_cores > 1`.')
            raise TaskError(msg)

    elif task['run_options']['num_cores'] <= 0:
        msg = 'Specify `num_cores` (in `run_options`) as an integer greater than 0.'
        raise TaskError(msg)

    elif 'pe' in task['run_options']:
        msg = ('No need to specify parallel environment (`pe`) in `run_options`, since '
               '`num_cores=1`.')
        raise TaskError(msg)

    # Make TaskSchema:
    if is_from_file:
        # Load from file (don't rely on the task schema existing on this installation):
        schema = TaskSchema(**task['schema'])

    else:
        # Find the software instance:
        soft_inst = get_software_instance(
            task.pop('software'),
            task['run_options']['num_cores'],
            all_software,
        )
        task['software_instance'] = soft_inst
        schema_dict = get_schema_dict(
            task['name'],
            task['method'],
            all_task_schemas,
            soft_inst
        )
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
            'local_input_names': list(i['local_inputs']['inputs'].keys()),
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


def set_default_nesting(task_lst, dep_idx):
    'Set default `nest` and `merge_priority` for each task.'

    for idx, downstream_tsk in enumerate(task_lst):

        # Do any further downstream tasks depend on this task?
        depended_on = False
        for dep_idx_i in dep_idx[(idx + 1):]:
            if idx in dep_idx_i:
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


def resolve_group(group, local_inputs, repeats_idx):

    # TODO (later): allow group_by on inputs from upstream tasks.

    combined_arr = []
    new_group_by = []
    for i in group['group_by']:
        if (i != 'repeats') and (i not in local_inputs):
            # Can only group on locally parametrised inputs.
            continue
        else:
            new_group_by.append(i)
        if i != 'repeats':
            combined_arr.append(local_inputs[i]['vals_idx'])
        else:
            combined_arr.append([(i or -1) for i in repeats_idx])

    if combined_arr:
        combined_arr = np.vstack(combined_arr)
        _, group_idx = np.unique(combined_arr, axis=1, return_inverse=True)
        group_elem_idx = [list(np.where(group_idx == i)[0])
                          for i in range(max(group_idx) + 1)]
        group_idx = list(group_idx)
    else:
        length = len(repeats_idx)
        group_idx = [0] * length
        group_elem_idx = [arange(length)]

    group_resolved = copy.deepcopy(group)
    group_resolved.update({
        'group_idx': group_idx,
        'group_element_idx': group_elem_idx,
        'num_groups': len(group_elem_idx),
        'group_size': len(group_elem_idx[0]),
        'group_by': new_group_by,
    })

    if 'merge_priority' not in group_resolved:
        group_resolved.update({'merge_priority': None})

    return group_resolved


def get_element_idx(task_lst, dep_idx):
    """For each task, find the element indices that determine the elements to be used
    (i.e from upstream tasks) to populate task inputs.

    Parameters
    ----------
    task_lst : list of dict
        Ordered task list. Each dict must have the following keys:
            name : str
            local_inputs : list
                List of dicts with keys:
                    length : int
                    repeats_idx : list
                    inputs : dict
                    groups : dict
            schema : TaskSchema
            task_idx : int

    dep_idx : list of list of int
        List of length equal to `task_lst`, whose elements are integer lists that link a
        given task to the indices of tasks upon which it depends.

    Returns
    -------
    element_idx : list of dict

    """

    # TODO: need to validate (somewhere) any specified merge_priority is >= 0?
    # TODO: check that local_inputs has an empty `inputs` dict when no local inputs
    # defined.

    # todo ensure default nest and merge_priority are set on each group (in local_inputs).

    element_idx = []
    for idx, downstream_task in enumerate(task_lst):

        upstream_tasks = [task_lst[i] for i in dep_idx[idx]]

        # local inputs dict:
        loc_in = downstream_task['local_inputs']
        schema = downstream_task['schema']

        if not upstream_tasks:
            # This task does not depend on any other tasks.
            groups = {}
            for group_name, group in loc_in['groups'].items():
                group = resolve_group(group, loc_in['inputs'], loc_in['repeats_idx'])
                groups.update({group_name: group})

            input_idx = arange(loc_in['length'])
            elem_idx_i = {
                'num_elements': loc_in['length'],
                'groups': groups,
                'inputs': {i: {'input_idx': input_idx} for i in loc_in['inputs']},
            }

        else:
            # This task depends on other tasks.
            ins_local = list(loc_in['inputs'].keys())
            ins_non_local = [i for i in schema.inputs if i['name'] not in ins_local]

            # Get the (local inputs) group dict for each `ins_non_local` (from upstream
            # tasks):
            input_groups = {}
            for non_loc_inp in ins_non_local:
                input_alias = non_loc_inp['alias']
                input_name = non_loc_inp['name']
                input_context = non_loc_inp['context']
                group_name = schema.get_input_by_alias(input_alias)['group']
                for up_task in upstream_tasks:
                    if input_context is not None:
                        if up_task['context'] != input_context:
                            continue
                    if input_name in up_task['schema'].outputs:
                        group_name_ = group_name
                        if group_name != 'default':
                            group_name_ = 'user_group_' + group_name
                        group_dict = element_idx[up_task['task_idx']]['groups']
                        group_names_fmt = ', '.join([f'"{i}"' for i in group_dict.keys()])
                        group_dat = group_dict.get(group_name_)
                        if group_dat is None:
                            msg = (f'No group "{group_name}" defined in the workflow for '
                                   f'input "{input_name}". Defined groups are: '
                                   f'{group_names_fmt}.')
                            raise UnsatisfiedGroupParameter(msg)
                        input_groups.update({
                            input_alias: {
                                **group_dat,
                                'group_name': group_name,
                                'task_idx': up_task['task_idx'],
                                'task_name': up_task['name'],
                            }
                        })
                        break

            is_nesting_mixed = len(set([i['nest'] for i in input_groups.values()])) > 1
            for input_alias, group_info in input_groups.items():

                if group_info['merge_priority'] is None and is_nesting_mixed:
                    raise MissingMergePriority(
                        f'`merge_priority` for group ("{group_info["group_name"]}") of '
                        f'input "{input_alias}" from task "{group_info["task_name"]}" must'
                        f' be specified because nesting is mixed. (Attempting to merge '
                        f'into task "{downstream_task["name"]}").'
                    )

                elif group_info['merge_priority'] is not None and not is_nesting_mixed:
                    warn(
                        f'`merge_priority` for group ("{group_info["group_name"]}") of '
                        f'input "{input_alias}" from task "{group_info["task_name"]}" is '
                        f'specified but not required because nesting is not mixed. '
                        f'(Merging into task "{downstream_task["name"]}").'
                    )

            all_mp = {k: (v['merge_priority'] or 0) for k, v in input_groups.items()}
            merging_order = [i[0] for i in sorted(all_mp.items(), key=lambda j: j[1])]

            # Cannot propagate groups if this task has elements that are sourced from
            # multiple upstream elements (group idx would be ill-defined). Keys are
            # task_idx:
            non_unit_group_sizes = {}

            ins_dict = {}
            groups = {}  # groups defined on the downstream task
            consumed_groups = []
            for idx, input_alias in enumerate(merging_order):

                in_group = input_groups[input_alias]
                if in_group['group_name'] != 'default':
                    consumed_groups.append('user_group_' + in_group['group_name'])

                incoming_size = in_group['num_groups']
                group_size = in_group['group_size']
                if group_size > 1:
                    non_unit_group_sizes.update({in_group['task_idx']: True})
                elif in_group['task_idx'] not in non_unit_group_sizes:
                    non_unit_group_sizes.update({in_group['task_idx']: False})

                if idx == 0:
                    if loc_in['inputs']:
                        existing_size = loc_in['length']
                        repeats_idx = loc_in['repeats_idx']
                        input_idx = arange(existing_size)
                        for i in loc_in['inputs']:
                            inp_alias = [j['alias'] for j in schema.inputs
                                         if j['name'] == i][0]
                            ins_dict.update({inp_alias: {'input_idx': input_idx}})

                        for group_name, group in loc_in['groups'].items():
                            group = resolve_group(group, loc_in['inputs'], repeats_idx)
                            groups.update({group_name: group})

                    else:
                        existing_size = incoming_size
                        repeats_idx = [None] * existing_size
                        ins_dict.update({
                            input_alias: {
                                'task_idx': in_group['task_idx'],
                                'group': in_group['group_name'],
                                'element_idx': in_group['group_element_idx'],
                            }
                        })
                        continue

                if in_group['nest']:

                    # Repeat existing:
                    for k, v in ins_dict.items():
                        if 'task_idx' in v:
                            elems_idx = repeat(v['element_idx'], incoming_size)
                            ins_dict[k]['element_idx'] = elems_idx
                        else:
                            input_idx = repeat(ins_dict[k]['input_idx'], incoming_size)
                            ins_dict[k]['input_idx'] = input_idx
                        repeats_idx = repeat(repeats_idx, incoming_size)

                    # Tile incoming:
                    ins_dict.update({
                        input_alias: {
                            'task_idx': in_group['task_idx'],
                            'group': in_group['group_name'],
                            'element_idx': tile(
                                in_group['group_element_idx'],
                                existing_size
                            ),
                        }
                    })

                    # Generate new groups for each group name:
                    for g_name, g in groups.items():

                        new_g_idx = extend_index_list(g['group_idx'], incoming_size)
                        new_ge_idx = to_sub_list(
                            extend_index_list(
                                flatten_list(g['group_element_idx']),
                                incoming_size),
                            g['group_size']
                        )
                        groups[g_name]['group_idx'] = new_g_idx
                        groups[g_name]['group_element_idx'] = new_ge_idx
                        groups[g_name]['num_groups'] = g['num_groups'] * incoming_size

                    existing_size *= incoming_size

                else:

                    if incoming_size != existing_size:
                        msg = (
                            f'Cannot merge input "{input_alias}" from task '
                            f'"{in_group["task_name"]}" and group '
                            f'"{in_group["group_name"]}" into task '
                            f'"{downstream_task["name"]}". Input has '
                            f'{incoming_size} elements, but current task has (at this '
                            f'point in the merge process) {existing_size} elements.'
                        )
                        raise IncompatibleTaskNesting(msg)

                    ins_dict.update({
                        input_alias: {
                            'task_idx': in_group['task_idx'],
                            'group': in_group['group_name'],
                            'element_idx': in_group['group_element_idx'],
                        }
                    })

            # Try to propagate into this task any non-default groups from dependent tasks:
            prop_groups = {}
            for up_task in upstream_tasks:
                for k, v in element_idx[up_task['task_idx']]['groups'].items():
                    if (
                        k != 'default' and
                        k not in groups and
                        k not in consumed_groups
                    ):
                        if not non_unit_group_sizes[up_task['task_idx']]:
                            prop_groups.update({k: v})
                        else:
                            msg = (
                                f'Cannot propagate group "{k}" from task '
                                f'"{up_task["name"]}" into task '
                                f'"{downstream_task["name"]}", because elements of task '
                                f'"{downstream_task["name"]}" are sourced from non-unit-'
                                f'sized element groups from task "{up_task["name"]}".'
                            )
                            raise IncompatibleWorkflow(msg)

            for group_name, g in prop_groups.items():

                group_reps = existing_size // (g['num_groups'] * g['group_size'])
                new_g_idx = extend_index_list(g['group_idx'], group_reps)
                new_ge_idx = to_sub_list(
                    extend_index_list(
                        flatten_list(g['group_element_idx']),
                        group_reps),
                    g['group_size']
                )
                prop_groups[group_name]['group_idx'] = new_g_idx
                prop_groups[group_name]['group_element_idx'] = new_ge_idx
                prop_groups[group_name]['num_groups'] = g['num_groups'] * group_reps

            all_groups = {**groups, **prop_groups}

            elem_idx_i = {
                'num_elements': existing_size,
                'inputs': ins_dict,
                'groups': all_groups,
            }

        element_idx.append(elem_idx_i)

    return element_idx


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
    task_lst = [
        validate_task_dict(i, is_from_file, SOFTWARE, TASK_SCHEMAS, check_integrity)
        for i in task_lst
    ]

    # Get dependencies, sort and add `task_idx` to each task:
    task_lst, dep_idx = order_tasks(task_lst)

    # Find element indices that determine the elements from which task inputs are drawn:
    element_idx = get_element_idx(task_lst, dep_idx)

    task_objs = [Task(**i) for i in task_lst]

    return task_objs, element_idx
