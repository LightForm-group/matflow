"""`matflow.models.task.py`

Module containing the Task and TaskSchema classes.

"""

import re
import secrets
from pprint import pprint

import numpy as np

from matflow.models import CommandGroup
from matflow.errors import TaskSchemaError, TaskParameterError


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

    @classmethod
    def load_from_hierarchy(cls, schema_lst):

        all_schema_dicts = {}
        for schema in schema_lst:

            name = schema['name']

            for method in schema['methods']:

                for imp in method['implementations']:

                    key = (name, method['name'], imp['name'])
                    if key in all_schema_dicts:
                        msg = (f'Schema with name "{name}", method "{method["name"]}" '
                               f'and implementation "{imp["name"]}" is multiply defined.')
                        raise ValueError(msg)

                    input_map = imp.get('input_map', [])
                    output_map = imp.get('output_map', [])
                    command_group = {'commands': imp.get('commands', [])}
                    all_inputs = (
                        schema.get('inputs', []) +
                        method.get('inputs', []) +
                        imp.get('inputs', [])
                    )
                    all_outputs = list(set(
                        schema.get('outputs', []) +
                        method.get('outputs', []) +
                        imp.get('outputs', [])
                    ))
                    all_schema_dicts.update({
                        key: {
                            'name': name,
                            'method': method['name'],
                            'implementation': imp['name'],
                            'inputs': all_inputs,
                            'outputs': all_outputs,
                            'input_map': input_map,
                            'output_map': output_map,
                            'command_group': command_group,
                        }
                    })

        all_schemas = {k: TaskSchema(**v) for k, v in all_schema_dicts.items()}

        return all_schemas

    @property
    def input_names(self):
        return [i['name'] for i in self.inputs]

    @property
    def input_aliases(self):
        return [i.get('alias', i['name']) for i in self.inputs]

    @property
    def input_contexts(self):
        return list(set([i.get('context', None) for i in self.inputs]))

    def _validate_inputs_outputs(self):
        'Basic checks on inputs and outputs.'

        allowed_inp_specifiers = ['group', 'context', 'alias']
        req_inp_keys = ['name']
        allowed_inp_keys = req_inp_keys + allowed_inp_specifiers
        allowed_inp_keys_fmt = ', '.join(['"{}"'.format(i) for i in allowed_inp_keys])

        err = (f'Validation failed for task schema "{self.name}" with method '
               f'"{self.method}" and software "{self.implementation}". ')

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
                        raise TaskSchemaError(err + msg)

            elif not isinstance(inp, dict):
                raise TypeError(err + 'Task schema input must be a str or a dict.')

            for r in req_inp_keys:
                if r not in inp:
                    msg = f'Task schema input must include key {r}.'
                    raise TaskSchemaError(err + msg)

            if 'context' not in inp:
                # Add default parameter context:
                inp['context'] = None

            if 'group' not in inp:
                # Add default group:
                inp['group'] = 'default'

            if 'alias' not in inp:
                # Add default alias:
                inp['alias'] = inp['name']

            unknown_inp_keys = list(set(inp.keys()) - set(allowed_inp_keys))
            if unknown_inp_keys:
                unknown_inp_keys_fmt = ', '.join([f'"{i}"' for i in unknown_inp_keys])
                msg = (f'Unknown task schema input keys: {unknown_inp_keys_fmt}. Allowed '
                       f'keys are: {allowed_inp_keys_fmt}.')
                raise TaskSchemaError(err + msg)

            self.inputs[inp_idx] = inp

        # Check the task does not output an input(!):
        for i in self.outputs:
            if i in self.input_names:
                msg = f'Task schema input "{i}" cannot also be an output!'
                raise TaskSchemaError(err + msg)

        # Check correct keys in supplied input/output maps:
        for in_map in self.input_map:
            if sorted(in_map.keys()) != sorted(['inputs', 'file']):
                bad_keys_fmt = ', '.join(['"{}"'.format(i) for i in in_map.keys()])
                msg = (f'Input maps must map a list of `inputs` into a `file` but found '
                       f'input map with keys {bad_keys_fmt}.')
                raise TaskSchemaError(err + msg)
            if not isinstance(in_map['inputs'], list):
                msg = 'Input map `inputs` must be a list.'
                raise TaskSchemaError(err + msg)

        for out_map in self.output_map:

            req_keys = ['files', 'output']
            allowed_keys = set(req_keys + ['options'])
            miss_keys = list(set(req_keys) - set(out_map.keys()))
            bad_keys = list(set(out_map.keys()) - allowed_keys)

            msg = (f'Output maps must map a list of `files` into an `output` (with '
                   f'optional `options`). ')
            if miss_keys:
                miss_keys_fmt = ', '.join(['"{}"'.format(i) for i in miss_keys])
                raise TaskSchemaError(err + msg + f'Missing keys are: {miss_keys_fmt}.')

            if bad_keys:
                bad_keys_fmt = ', '.join(['"{}"'.format(i) for i in bad_keys])
                raise TaskSchemaError(err + msg + f'Unknown keys are: {bad_keys_fmt}.')

            if not isinstance(out_map['output'], str):
                msg = 'Output map `output` must be a string.'
                raise TaskSchemaError(err + msg)

            for i in out_map['files']:
                if ('name' not in i) or ('save' not in i):
                    msg = (f'Specify keys `name` (str) and `save` (bool) in output map '
                           f'`files` key.')
                    raise TaskSchemaError(err + msg)

        # Check inputs/outputs named in input/output_maps are in inputs/outputs lists:
        input_map_ins = [j for i in self.input_map for j in i['inputs']]
        unknown_map_inputs = set(input_map_ins) - set(self.input_aliases)

        output_map_outs = [i['output'] for i in self.output_map]
        unknown_map_outputs = set(output_map_outs) - set(self.outputs)

        if unknown_map_inputs:
            bad_ins_map_fmt = ', '.join(['"{}"'.format(i) for i in unknown_map_inputs])
            msg = (f'Input map inputs {bad_ins_map_fmt} not known by the schema with '
                   f'input (aliases): {self.input_aliases}.')
            raise TaskSchemaError(err + msg)

        if unknown_map_outputs:
            bad_outs_map_fmt = ', '.join(['"{}"'.format(i) for i in unknown_map_outputs])
            msg = (f'Output map outputs {bad_outs_map_fmt} not known by the schema with '
                   f'outputs: {self.outputs}.')
            raise TaskSchemaError(err + msg)

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

    def check_output_map_options(self, options):
        'Check a set of options are consistent with the output map options.'

        req_opts, opt_opts = [], []
        for i in self.output_map:
            out_map_opts = i.get('options', {})
            req_opts.extend(out_map_opts.get('required', []))
            opt_opts.extend(out_map_opts.get('optional', []))

        miss_opts = list(set(req_opts) - set(options))
        if miss_opts:
            miss_opts_fmt = ', '.join([f'"{i}"' for i in miss_opts])
            msg = (f'Output maps for the schema "{self.name}" have the following '
                   f'required output map options that are not specified in the task: '
                   f'{miss_opts_fmt}.')
            raise TaskParameterError(msg)

        bad_opts = list(set(options) - set(req_opts + opt_opts))
        if bad_opts:
            bad_opts_fmt = ', '.join([f'"{i}"' for i in bad_opts])
            msg = (f'Output maps for the schema "{self.name}" are not compatible with '
                   f'the following output map options that are specified in the task: '
                   f'{bad_opts_fmt}.')
            raise TaskParameterError(msg)

    @property
    def is_func(self):
        return not self.command_group.commands

    def __repr__(self):
        out = (
            f'{self.__class__.__name__}('
            f'name={self.name!r}, method={self.method!r}, '
            f'inputs={self.inputs!r}, outputs={self.outputs!r})'
        )
        return out

    def get_input_by_name(self, input_name):
        for i in self.inputs:
            if i['name'] == input_name:
                return i
        raise ValueError(f'No input "{input_name}" in schema.')

    def get_input_by_alias(self, input_alias):
        for i in self.inputs:
            if i['alias'] == input_alias:
                return i
        raise ValueError(f'No input alias "{input_alias}" in schema.')


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
        '_id',
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
        '_output_map_options',
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
                 merge_priority=None, output_map_options=None):

        self._id = None  # Generated once by generate_id()

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
        self._output_map_options = output_map_options
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

    def as_dict(self):
        'Return attributes dict with preceding underscores removed.'
        return {k.lstrip('_'): getattr(self, k) for k in self.__slots__}

    def generate_id(self):
        self.id = secrets.token_hex(10)

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, id_):
        if self._id:
            raise ValueError(f'ID is already set for Task. ID is: "{self.id}".')
        else:
            self._id = id_

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

    @status.setter
    def status(self, status):
        'Set task status'
        # TODO validate, maybe with enum.
        self._status = status

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

    @inputs.setter
    def inputs(self, inputs):
        'Set the task inputs (i.e. from `Workflow.prepare_task`).'
        if not isinstance(inputs, list) or not isinstance(inputs[0], dict):
            raise ValueError('Inputs must be a list of dict.')
        self._inputs = inputs

    @property
    def outputs(self):
        return self._outputs

    @outputs.setter
    def outputs(self, outputs):
        'Set the task outputs (i.e. from `Workflow.process_task`).'
        if not isinstance(outputs, list) or not isinstance(outputs[0], dict):
            raise ValueError('Outputs must be a list of dict.')
        self._outputs = outputs

    @property
    def output_map_options(self):
        return self._output_map_options

    @property
    def schema(self):
        return self._schema

    @property
    def files(self):
        return self._files

    @files.setter
    def files(self, files):
        'Set the task files (i.e. from `Workflow.prepare_task`).'
        if not isinstance(files, list) or not isinstance(files[0], dict):
            raise ValueError('Files must be a list of dict.')
        self._files = files

    @property
    def resource_usage(self):
        return self._resource_usage

    @resource_usage.setter
    def resource_usage(self, resource_usage):
        self._resource_usage = resource_usage

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
