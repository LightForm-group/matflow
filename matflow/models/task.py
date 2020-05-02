"""`matflow.models.task.py`"""

import copy
import re
from pathlib import Path
from warnings import warn
from pprint import pprint

import numpy as np

from matflow import CONFIG, CURRENT_MACHINE, SOFTWARE, TASK_SCHEMAS
from matflow.models import CommandGroup, Command
from matflow.jsonable import to_jsonable
from matflow.utils import parse_times, nest_lists, combine_list_of_dicts
from matflow.errors import IncompatibleSequence, TaskSchemaError, TaskParameterError


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


def resolve_local_inputs(base=None, num_repeats=None, sequences=None):
    """Transform `base` and `sequences` into `input` list."""

    # TODO: delete this function.

    # print('\nbase:')
    # pprint(base)

    # print('sequences:')
    # pprint(sequences)

    if num_repeats is not None and sequences is not None:
        raise ValueError('Specify one of `num_repeats` of `sequences`.')

    if base is None:
        base = {}

    if num_repeats:
        local_inputs = [base for _ in range(num_repeats)]
    else:
        local_inputs = [base]

    if sequences is not None:

        # === Don't modify original:
        sequences = copy.deepcopy(sequences)

        # === Check equal `nest_idx` sequences have the same number of `vals`
        num_vals_map = {}
        for seq in sequences:
            prev_num_vals = num_vals_map.get(seq['nest_idx'])
            cur_num_vals = len(seq['vals'])

            # print('\tprev_num_vals: {}'.format(prev_num_vals))

            if prev_num_vals is None:
                num_vals_map.update({seq['nest_idx']: cur_num_vals})

            elif prev_num_vals != cur_num_vals:
                msg = ('Sequences with the same `nest_idx` must have the same number of '
                       'values.')
                raise IncompatibleSequence(msg)

            # print('\tcur_num_vals: {}'.format(cur_num_vals))

        # print('Now sorting the sequences by nest_idx.')

        # === Sort by `nest_idx`
        sequences.sort(key=lambda x: x['nest_idx'])

        # print('Sorted sequences:')
        # pprint(sequences)

        # === Turn `vals` into list of dicts
        for seq_idx, seq in enumerate(sequences):
            sequences[seq_idx]['vals'] = [{seq['name']: i} for i in seq['vals']]

        # print('Turning vals into list of dict. Sequences now:')
        # pprint(sequences)

        local_inputs = combine_base_sequence(sequences, base)

    print('\ntask.resolve_local_inputs: local_inputs:')
    pprint(local_inputs)

    return local_inputs


def combine_base_sequence(sequences, base=None):

    # TODO: delete this function.

    # print('combine_base_sequence: sequences:')
    # pprint(sequences)

    if base is None:
        base = {}

    # Merge parallel sequences:
    merged_seqs = []
    skip_idx = []
    for seq_idx, seq in enumerate(sequences):

        if seq_idx in skip_idx:
            continue

        merged_seqs.append(seq)
        merged_seqs[-1]['name'] = [merged_seqs[-1]['name']]

        for next_idx in range(seq_idx + 1, len(sequences)):

            if sequences[next_idx]['nest_idx'] == seq['nest_idx']:

                # Merge values:
                for val_idx in range(len(merged_seqs[-1]['vals'])):
                    merged_seqs[-1]['vals'][val_idx].update(
                        sequences[next_idx]['vals'][val_idx]
                    )
                # Merge names:
                name_old = merged_seqs[-1]['name']
                merged_seqs[-1]['name'] += [sequences[next_idx]['name']]

                skip_idx.append(next_idx)

    # Nest nested sequences:
    seq_vals = [i['vals'] for i in merged_seqs]
    nested_seqs = nest_lists(seq_vals)

    nested_seq_all = []
    for seq in nested_seqs:
        nested_seq_all.append(combine_list_of_dicts([base] + seq))

    return nested_seq_all


def normalise_local_inputs(base=None, sequences=None):
    'Validate and normalise task inputs.'

    if base is None:
        base = {}
    if sequences is None:
        sequences = []

    nest_req = True if len(sequences) > 1 else False

    prev_num_vals = None
    prev_nest = None
    inputs_lst = []
    for seq in sequences:

        if 'name' not in seq:
            msg = '`name` key is required for sequence.'
            raise ValueError(msg)

        if 'vals' not in seq:
            msg = '`vals` is required for sequence "{}"'
            raise ValueError(msg.format(seq['name']))
        else:
            num_vals = len(seq['vals'])

        if nest_req:
            if 'nest_idx' not in seq:
                msg = '`nest_idx` is required for sequence "{}"'
                raise ValueError(msg.format(seq['name']))
            else:
                if seq['nest_idx'] < 0:
                    msg = '`nest_idx` must be a positive integer or zero for sequence "{}"'
                    raise ValueError(msg.format(seq['name']))
        else:
            # Set a default `nest_idx`:
            seq['nest_idx'] = 0

        nest = seq['nest_idx']

        if prev_num_vals and nest == prev_nest:
            # For same nest_idx, sequences must have the same lengths:
            if num_vals != prev_num_vals:
                msg = ('Sequence "{}" shares a `nest_idx` with another sequence but has '
                       'a different number of values.')
                raise ValueError(msg.format(seq['name']))

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


def get_local_inputs(base=None, repeats=None, sequences=None, groups=None):

    inputs_lst = normalise_local_inputs(base, sequences)

    if repeats is None:
        repeats = 1

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

    inputs_dct = {'inputs': {}}

    for idx, input_i in enumerate(inputs_lst):

        if (prev_nest is None) or (input_i['nest_idx'] > prev_nest):
            rep_i = int(prev_reps / lengths[idx])
            tile_i = int(total_len / (lengths[idx] * rep_i))
            prev_nest = input_i['nest_idx']
        else:
            rep_i = prev_reps
            tile_i = prev_tile

        vals_idx = np.tile(np.repeat(np.arange(lengths[idx]), rep_i), tile_i)
        vals_idx = np.repeat(vals_idx, repeats)
        repeats_idx = np.tile(np.arange(repeats), (total_len,))

        inputs_dct['repeats_idx'] = repeats_idx
        inputs_dct['inputs'].update({
            input_i['name']: {
                'nest_idx': input_i['nest_idx'],
                'vals': input_i['vals'],
                'vals_idx': vals_idx,
            }
        })
        prev_reps = rep_i
        prev_tile = tile_i

    group_idx = {}
    if groups:

        allowed_grp = list(inputs_dct['inputs'].keys()) + ['repeats']
        allowed_grp_fmt = ', '.join([f'"{i}"' for i in allowed_grp])

        for group_name, group_params_lst in groups.items():

            if not group_params_lst:
                group_idx.update({group_name: np.zeros(total_len, dtype=int)})

            else:

                for param_name in group_params_lst:
                    if param_name not in allowed_grp:
                        msg = (f'Parameter "{param_name}" cannot be grouped, because it '
                               f'has no specified values. Allowed group values are: '
                               f'{allowed_grp_fmt}.')
                        raise ValueError(msg)

                combined_arr = []
                for i in group_params_lst:
                    if i != 'repeats':
                        combined_arr.append(inputs_dct['inputs'][i]['vals_idx'])
                    else:
                        combined_arr.append(inputs_dct['repeats_idx'])

                combined_arr = np.vstack(combined_arr)
                _, group_i_idx = np.unique(combined_arr, axis=1, return_inverse=True)
                group_idx.update({group_name: group_i_idx})

    total_len *= repeats
    inputs_dct.update({
        'length': total_len,
        'group_idx': group_idx,
    })

    return inputs_dct


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

    INIT_STATUS = 'pending'

    def __init__(self, name, method, software_instance, task_idx, nest=None,
                 merge_priority=None, run_options=None, base=None, sequences=None,
                 repeats=None, local_inputs=None, inputs=None, outputs=None,
                 schema=None, status=None, pause=False, files=None, resource_usage=None,
                 stats=True, context=''):

        self.name = name
        self.status = status or Task.INIT_STATUS  # | 'paused' | 'complete'
        self.method = method
        self.task_idx = task_idx
        self.nest = nest
        self.merge_priority = merge_priority
        self.software_instance = software_instance
        self.run_options = run_options
        self.context = context
        self.local_inputs = local_inputs
        self.inputs = inputs
        self.outputs = outputs
        self.pause = pause
        self.schema = schema
        self.files = files
        self.resource_usage = resource_usage
        self.stats = stats

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
    def name_friendly(self):
        'Capitalise and remove underscores'
        name = '{}{}'.format(self.name[0].upper(), self.name[1:]).replace('_', ' ')
        return name

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

    def initialise_outputs(self):
        self.outputs = [
            {key: None for key in self.schema.outputs}
            for _ in range(len(self.local_inputs))
        ]
