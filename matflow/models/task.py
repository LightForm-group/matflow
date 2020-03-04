"""`matflow.models.task.py`"""

import copy
from pathlib import Path
from warnings import warn
from pprint import pprint

import numpy as np

from matflow import CONFIG, CURRENT_MACHINE, SOFTWARE, TASK_SCHEMAS
from matflow.models import CommandGroup, Command
from matflow.jsonable import to_jsonable
from matflow.sequence import combine_base_sequence
from matflow.utils import parse_times
from matflow.errors import (IncompatibleWorkflow, IncompatibleNesting,
                            MissingMergePriority)


def check_task_compatibility(task_compat_props):
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

    dependency_idx = []
    all_outputs = []
    for ins_outs_i in task_compat_props:
        all_outputs.extend(ins_outs_i['outputs'])
        output_idx = []
        for input_j in ins_outs_i['inputs']:
            for task_idx_k, ins_outs_k in enumerate(task_compat_props):
                if input_j in ins_outs_k['outputs']:
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

    # Find the minimum index at which each task must be positioned to satisfy input
    # dependencies:
    min_idx = [max(i or [0]) + 1 for i in dependency_idx]
    task_srt_idx = np.argsort(min_idx)

    # Reorder:
    task_compat_props = [task_compat_props[i] for i in task_srt_idx]
    dependency_idx = [[np.argsort(task_srt_idx)[j] for j in dependency_idx[i]]
                      for i in task_srt_idx]

    # Note: when considering upstream tasks for a given downstream task, need to nest
    # according to the upstream tasks' `num_elements`, not their `length`.
    inputs_idx = []
    for idx, downstream_task in enumerate(task_compat_props):

        # Do any further downstream tasks depend on this task?
        depended_on = False
        for deps_idx in dependency_idx[(idx + 1):]:
            if idx in deps_idx:
                depended_on = True
                break

        if not depended_on and downstream_task.get('nest') is not None:
            msg = '`nest` value is specified but not required for task "{}".'.format(
                downstream_task['name'])
            warn(msg)

        # Add `nest: True` by default to this task if nesting is not specified, and if at
        # least one further downstream task depends on this task:
        if downstream_task.get('nest', None) is None:
            if depended_on:
                downstream_task['nest'] = True

        # Add default `merge_priority` of `None`:
        if 'merge_priority' not in downstream_task:
            downstream_task['merge_priority'] = None

        # Find num_elements on downstream task:
        if not dependency_idx[idx]:
            num_elements = downstream_task['length']

        else:
            upstream_tasks = [task_compat_props[i] for i in dependency_idx[idx]]
            is_nesting_mixed = len(set([i['nest'] for i in upstream_tasks])) > 1

            for i in upstream_tasks:

                if i['merge_priority'] is None and is_nesting_mixed:
                    msg = ('`merge_priority` for task "{}" must be specified, because'
                           ' nesting is mixed.'.format(i['name']))
                    raise MissingMergePriority(msg)

                elif i['merge_priority'] is not None and not is_nesting_mixed:
                    msg = ('`merge_priority` value is specified but not required for '
                           'task "{}", because nesting is not mixed.'.format(
                               downstream_task['name']))
                    warn(msg)

            all_merge_priority = [i.get('merge_priority') or 0 for i in upstream_tasks]
            merging_order = np.argsort(all_merge_priority)

            num_elements = downstream_task['length']
            inputs_idx = {}
            for input_name in downstream_task['inputs']:
                inputs_idx.update({
                    input_name: {
                        'task_idx': -1,
                        'element_idx': list(range(num_elements)),
                    }
                })

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
                        raise IncompatibleNesting(msg)

        downstream_task['num_elements'] = num_elements

    return list(task_srt_idx), task_compat_props


class TaskSchema(object):
    """Class to represent the schema of a particular method/implementation of a task.

            'name': self.name,
            'method': self.method,
            'implementation': self.software,
            'inputs': inputs,
            'outputs': outputs,
            'input_map': in_map,
            'output_map': out_map,
            'command_group': command_group,
    """

    def __init__(self, name, method=None, implementation=None, inputs=None,
                 outputs=None, input_map=None, output_map=None,
                 command_group=None):
        """Instantiate a TaskSchema.

        to check:
        *   all inputs/outputs referred to in the input/output_maps are also in the
            input/output lists.

        """
        self.name = name
        self.method = method
        self.implementation = implementation
        self.inputs = inputs
        self.outputs = outputs
        self.input_map = input_map
        self.output_map = output_map

        self.command_group = CommandGroup(**command_group) if command_group else None

    @property
    def is_func(self):
        return not self.implementation

    def __repr__(self):
        out = (
            f'{self.__class__.__name__}('
            f'name={self.name!r}, method={self.method!r}, '
            f'inputs={self.inputs!r}, outputs={self.outputs!r})'
        )
        return out


class Task(object):

    INIT_STATUS = 'pending'

    def __init__(self, name, method, software_instance, task_idx, nest=None,
                 merge_priority=None, run_options=None, base=None, sequences=None,
                 num_repeats=None, inputs_local=None, outputs=None, schema=None, status=None,
                 pause=False):

        self.name = name
        self.status = status or Task.INIT_STATUS  # | 'paused' | 'complete'
        self.method = method
        self.task_idx = task_idx
        self.nest = nest
        self.merge_priority = merge_priority
        self.software_instance = software_instance
        self.run_options = run_options
        self.inputs_local = inputs_local
        self.outputs = outputs
        self.pause = pause

        self.schema = TaskSchema(**(schema or self._get_schema_dict()))

        if not self.inputs_local:
            self.inputs_local = self._resolve_inputs_local(base, num_repeats, sequences)

    def __repr__(self):
        out = (
            f'{self.__class__.__name__}('
            f'name={self.name!r}, '
            f'status={self.status!r}, '
            f'method={self.method!r}, '
            f'software_instance={self.software_instance!r}, '
            f'run_options={self.run_options!r}, '
            f'schema={self.schema!r}'
            f')'
        )
        return out

    def __len__(self):
        return len(self.inputs_local)

    def _resolve_inputs_local(self, base, num_repeats, sequences):
        """Transform `base` and `sequences` into `input` list."""

        if num_repeats is not None and sequences is not None:
            raise ValueError('Specify one of `num_repeats` of `sequences`.')

        # print('Task._resolve_inputs: ')

        # print('base')
        # pprint(base)

        # print('num_repeats')
        # pprint(num_repeats)

        # print('sequences')
        # pprint(sequences)

        # print('self.schema')
        # pprint(self.schema)

        if base is None:
            base = {}

        if num_repeats:
            out = [base for _ in range(num_repeats)]
        else:
            out = [base]

        if sequences is not None:
            # Don't modify original:
            sequences = copy.deepcopy(sequences)

            # Check equal `nest_idx` sequences have the same number of `vals`
            num_vals_map = {}
            for seq in sequences:
                # print('seq: ')
                # pprint(seq)
                prev_num_vals = num_vals_map.get(seq['nest_idx'])
                cur_num_vals = len(seq['vals'])
                if prev_num_vals is None:
                    num_vals_map.update({seq['nest_idx']: cur_num_vals})
                elif prev_num_vals != cur_num_vals:
                    raise ValueError(
                        'Sequences with the same `nest_idx` must '
                        'have the same number of values.'
                    )

            # Sort by `nest_idx`
            sequences.sort(key=lambda x: x['nest_idx'])

            # Turn `vals` into list of dicts
            for seq_idx, seq in enumerate(sequences):
                sequences[seq_idx]['vals'] = [
                    {seq['name']: i} for i in seq['vals']]

            out = combine_base_sequence(sequences, base)

        # print('out')
        # pprint(out)

        return out

    @property
    def software(self):
        return self.software_instance['name']

    @property
    def is_scheduled(self):
        if self.schema.is_func:
            return False
        else:
            return self.software_instance['scheduler'] != 'direct'

    def is_remote(self, workflow):
        return self.resource_name != workflow.resource_name

    @property
    def resource_name(self):
        return self.run_options['resource']

    def get_task_path(self, workflow_path):
        return workflow_path.joinpath(f'task_{self.task_idx}_{self.name}')

    def get_resource(self, workflow):
        'Get the Resource associated with this Task.'
        return workflow.resources[self.resource_name]

    def get_resource_conn(self, workflow):
        'Get the ResourceConnection associated with this Task.'
        src_name = workflow.resource_name
        dst_name = self.resource_name
        return workflow.resource_conns[(src_name, dst_name)]

    def _get_schema_dict(self):
        """Get the schema associated with the method/implementation of this task."""

        match_task_idx = None
        match_method_idx = None
        match_imp_idx = None

        for task_ref_idx, task_ref in enumerate(TASK_SCHEMAS):

            if task_ref['name'] == self.name:

                match_task_idx = task_ref_idx
                for met_idx, met in enumerate(task_ref['methods']):

                    if met['name'] == self.method:

                        match_method_idx = met_idx
                        implementations = met.get('implementations')
                        if implementations:

                            for imp_idx, imp in enumerate(implementations):

                                if imp['name'] == self.software_instance['name']:
                                    match_imp_idx = imp_idx
                                    break
                        break
                break

        if match_task_idx is None:
            msg = (f'No matching task found with name: "{self.name}"')
            raise ValueError(msg)

        if match_method_idx is None:
            msg = (f'No matching method found with name: "{self.method}"'
                   f' in task: "{self.name}""')
            raise ValueError(msg)

        task_ref = TASK_SCHEMAS[match_task_idx]
        met_ref = task_ref['methods'][met_idx]
        inputs = task_ref.get('inputs', []) + met_ref.get('inputs', [])
        outputs = task_ref.get('outputs', []) + met_ref.get('outputs', [])

        imp_ref = None
        in_map = None
        out_map = None
        # command_group = None
        command_opt = None

        if match_imp_idx is not None:
            imp_ref = met_ref['implementations'][match_imp_idx]

            inputs += imp_ref.get('inputs', [])
            outputs += imp_ref.get('outputs', [])

            in_map = imp_ref.get('input_map', [])
            out_map = imp_ref.get('output_map', [])
            command_opt = imp_ref.get('commands', [])

            # commands = [Command(**i) for i in command_opt]
            # command_group = CommandGroup(
            #     commands, self.software_instance.get('env_pre'),
            #     self.software_instance.get('env_post')
            # )

        if self.software_instance:
            implementation = self.software_instance['name']
            command_group = {
                'commands': command_opt,
                'env_pre': self.software_instance.get('env_pre'),
                'env_post': self.software_instance.get('env_post'),
            }
        else:
            implementation = None
            command_group = None

        schema_dict = {
            'name': self.name,
            'method': self.method,
            'implementation': implementation,
            'inputs': inputs,
            'outputs': outputs,
            'input_map': in_map,
            'output_map': out_map,
            'command_group': command_group,
        }

        return schema_dict

    def initialise_outputs(self):
        self.outputs = [
            {key: None for key in self.schema.outputs}
            for _ in range(len(self.inputs_local))
        ]
