"""`matflow.models.task.py`"""

import copy
import re
from pathlib import Path
from warnings import warn
from pprint import pprint

import numpy as np

from matflow import CONFIG, TASK_SCHEMAS
from matflow.models import CommandGroup, Command
from matflow.jsonable import to_jsonable
from matflow.utils import tile, repeat, arange
from matflow.errors import (
    SequenceError,
    IncompatibleSequence,
    TaskSchemaError,
    TaskParameterError
)


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

    allowed_grp = list(schema_inputs.keys()) + ['repeats']
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

    def __init__(self, name, outputs, method=None, implementation=None, inputs=None,
                 input_map=None, output_map=None, command_group=None):
        'Instantiate a TaskSchema.'

        self.name = name
        self.outputs = outputs
        self.method = method
        self.implementation = implementation
        self.inputs = inputs or []
        self.input_map = input_map or []
        self.output_map = output_map or []

        self.command_group = CommandGroup(**command_group) if command_group else None

        self._validate_inputs_outputs()

    @property
    def input_names(self):
        return [i['name'] for i in self.inputs]

    @property
    def input_aliases(self):
        return [i.get('alias', i['name']) for i in self.inputs]

    @property
    def input_contexts(self):
        return list(set([i.get('context', '') for i in self.inputs]))

    def _validate_inputs_outputs(self):
        'Basic checks on inputs and outputs.'

        allowed_inp_specifiers = ['group', 'context', 'alias']
        req_inp_keys = ['name']
        allowed_inp_keys = req_inp_keys + allowed_inp_specifiers
        allowed_inp_keys_fmt = ', '.join(['"{}"'.format(i) for i in allowed_inp_keys])

        # Normalise schema inputs:
        for inp_idx, inp in enumerate(self.inputs):

            if isinstance(inp, str):

                # Parse additional input specifiers:
                match = re.search(r'([\w-]+)(\[(.*?)\])*', inp)
                inp_name = match.group(1)
                inp = {'name': inp_name}

                specifiers_str = match.group(3)
                if specifiers_str:
                    specs = specifiers_str.split(',')
                    for s in specs:
                        s_key, s_val = s.split('=')
                        inp.update({s_key.strip(): s_val.strip()})

                    if 'context' in inp and inp['context'] and 'alias' not in inp:
                        msg = ('Task schema inputs for which a `context` is specified '
                               'must also be given an `alias`.')
                        raise TaskSchemaError(msg)

            elif not isinstance(inp, dict):
                raise TypeError('Task schema input must be a str or a dict.')

            for r in req_inp_keys:
                if r not in inp:
                    msg = f'Task schema input must include key {r}.'
                    raise TaskSchemaError(msg)

            if 'context' not in inp:
                # Add default parameter context:
                inp['context'] = None

            unknown_inp_keys = set(inp.keys()) - set(allowed_inp_keys)
            if unknown_inp_keys:
                msg = (f'Unknown task schema input key: {unknown_inp_keys}. Allowed keys '
                       f'are: {allowed_inp_keys_fmt}')
                raise TaskSchemaError(msg)

            self.inputs[inp_idx] = inp

        # Check the task does not output an input(!):
        for i in self.outputs:
            if i in self.input_names:
                msg = 'Task schema input "{}" cannot also be an output!'
                raise TaskSchemaError(msg.format(i))

        # Check correct keys in supplied input/output maps:
        for in_map in self.input_map:
            if list(in_map.keys()) != ['inputs', 'file']:
                bad_keys_fmt = ', '.join(['"{}"'.format(i) for i in in_map.keys()])
                msg = ('Input maps must map a list of `inputs` into a `file` but found '
                       'input map with keys {} for schema "{}".')
                raise TaskSchemaError(msg.format(bad_keys_fmt, self.name))
            if not isinstance(in_map['inputs'], list):
                msg = 'Input map `inputs` must be a list for schema "{}".'
                raise TaskSchemaError(msg.format(self.name))

        for out_map in self.output_map:
            if list(out_map.keys()) != ['files', 'output']:
                bad_keys_fmt = ', '.join(['"{}"'.format(i) for i in out_map.keys()])
                msg = ('Output maps must map a list of `files` into an `output` but found '
                       'output map with keys {} for schema "{}".')
                raise TaskSchemaError(msg.format(bad_keys_fmt, self.name))
            if not isinstance(out_map['output'], str):
                msg = 'Output map `output` must be a string for schema "{}".'
                raise TaskSchemaError(msg.format(self.name))

        # Check inputs/outputs named in input/output_maps are in inputs/outputs lists:
        input_map_ins = [j for i in self.input_map for j in i['inputs']]
        unknown_map_inputs = set(input_map_ins) - set(self.input_aliases)

        output_map_outs = [i['output'] for i in self.output_map]
        unknown_map_outputs = set(output_map_outs) - set(self.outputs)

        if unknown_map_inputs:
            bad_ins_map_fmt = ', '.join(['"{}"'.format(i) for i in unknown_map_inputs])
            msg = ('Input map inputs {} not known by the schema "{}" with input '
                   '(aliases): {}.')
            raise TaskSchemaError(msg.format(
                bad_ins_map_fmt, self.name, self.input_aliases))

        if unknown_map_outputs:
            bad_outs_map_fmt = ', '.join(['"{}"'.format(i) for i in unknown_map_outputs])
            msg = 'Output map outputs {} not known by the schema "{}" with outputs: {}.'
            raise TaskSchemaError(msg.format(bad_outs_map_fmt, self.name, self.outputs))

    def check_surplus_inputs(self, inputs):
        'Check for any inputs that are specified but not required by this schema.'

        surplus_ins = set(inputs) - set(self.input_names)
        if surplus_ins:
            surplus_ins_fmt = ', '.join(['"{}"'.format(i) for i in surplus_ins])
            msg = 'Input(s) {} not known by the schema "{}" with inputs: {}.'
            raise TaskParameterError(msg.format(
                surplus_ins_fmt, self.name, self.input_names))

    def check_missing_inputs(self, inputs):
        'Check for any inputs that are required by this schema but not specified.'

        missing_ins = set(self.input_names) - set(inputs)
        if missing_ins:
            missing_ins_fmt = ', '.join(['"{}"'.format(i) for i in missing_ins])
            msg = 'Input(s) {} missing for the schema "{}" with inputs: {}'
            raise TaskParameterError(msg.format(
                missing_ins_fmt, self.name, self.input_names))

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
    """

    Notes
    -----
    As with `Workflow`, this class is "locked down" quite tightly by using `__slots__` and
    properties. This is to help with maintaining integrity of the workflow between
    save/load cycles.

    """

    INIT_STATUS = 'pending'

    __slots__ = [
        '_name',
        '_method',
        '_software_instance',
        '_task_idx',
        '_run_options',
        '_status',
        '_stats',
        '_context',
        '_local_inputs',
        '_inputs',
        '_outputs',
        '_schema',
        '_files',
        '_resource_usage',
        '_base',
        '_sequences',
        '_repeats',
        '_groups',
        '_nest',
        '_merge_priority',
    ]

    def __init__(self, name, method, software_instance, task_idx, run_options=None,
                 status=None, stats=True, context='', local_inputs=None, inputs=None,
                 outputs=None, schema=None, files=None, resource_usage=None,
                 base=None, sequences=None, repeats=None, groups=None, nest=None,
                 merge_priority=None):

        self._name = name
        self._method = method
        self._software_instance = software_instance
        self._task_idx = task_idx
        self._run_options = run_options
        self._status = status or Task.INIT_STATUS  # | 'paused' | 'complete'
        self._stats = stats
        self._context = context
        self._local_inputs = local_inputs
        self._inputs = inputs
        self._outputs = outputs
        self._schema = schema
        self._files = files
        self._resource_usage = resource_usage

        # Saved for completeness, and to allow regeneration of `local_inputs`:
        self._base = base
        self._sequences = sequences
        self._repeats = repeats
        self._groups = groups
        self._nest = nest
        self._merge_priority = merge_priority

    def __repr__(self):
        out = (
            f'{self.__class__.__name__}('
            f'name={self.name!r}, '
            f'status={self.status!r}, '
            f'method={self.method!r}, '
            f'software_instance={self.software_instance!r}, '
            f'run_options={self.run_options!r}, '
            f'context={self.context!r}, '
            f'schema={self.schema!r}'
            f')'
        )
        return out

    def __len__(self):
        return self.local_inputs['length']

    @property
    def name(self):
        return self._name

    @property
    def method(self):
        return self._method

    @property
    def software_instance(self):
        return self._software_instance

    @property
    def task_idx(self):
        return self._task_idx

    @property
    def run_options(self):
        return self._run_options

    @property
    def status(self):
        return self._status

    @property
    def stats(self):
        return self._stats

    @property
    def context(self):
        return self._context

    @property
    def local_inputs(self):
        return self._local_inputs

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def schema(self):
        return self._schema

    @property
    def files(self):
        return self._files

    @property
    def resource_usage(self):
        return self._resource_usage

    @property
    def base(self):
        return self._base

    @property
    def sequences(self):
        return self._sequences

    @property
    def repeats(self):
        return self._repeats

    @property
    def groups(self):
        return self._groups

    @property
    def nest(self):
        return self._nest

    @property
    def merge_priority(self):
        return self._merge_priority

    @property
    def name_friendly(self):
        'Capitalise and remove underscores'
        name = '{}{}'.format(self.name[0].upper(), self.name[1:]).replace('_', ' ')
        return name

    @property
    def software(self):
        return self.software_instance['name']

    def get_task_path(self, workflow_path):
        return workflow_path.joinpath(f'task_{self.task_idx}_{self.name}')
