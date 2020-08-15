"""`matflow.models.task.py`

Module containing the Task and TaskSchema classes.

"""

import copy
import enum
import re
import secrets
from pprint import pprint

import numpy as np

from matflow.models import CommandGroup
from matflow.errors import TaskSchemaError, TaskParameterError
from matflow.hicklable import to_hicklable
from matflow.utils import dump_to_yaml_string, get_specifier_dict
from matflow.models.software import SoftwareInstance
from matflow.models.element import Element


class TaskSchema(object):
    """Class to represent the schema of a particular method/implementation of a task."""

    def __init__(self, name, outputs, method=None, implementation=None, inputs=None,
                 input_map=None, output_map=None, command_group=None,
                 archive_excludes=None):
        'Instantiate a TaskSchema.'

        self.name = name
        self.outputs = outputs
        self.method = method
        self.implementation = implementation
        self.inputs = inputs or []
        self.input_map = input_map or []
        self.output_map = output_map or []
        self.archive_excludes = archive_excludes

        self.command_group = CommandGroup(**command_group) if command_group else None

        self._validate_inputs_outputs()
        self.command_group.check_pathway_conditions(self.input_names)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'name={self.name!r}, '
            f'method={self.method!r}, '
            f'implementation={self.implementation!r}, '
            f'inputs={self.inputs_condensed!r}, '
            f'outputs={self.outputs!r}, '
            f'command_group={self.command_group!r}, '
            f'archive_excludes={self.archive_excludes!r}'
            f')'
        )

    def __str__(self):
        self_dict = self.as_dict()
        self_dict['inputs'] = self.inputs_condensed
        return dump_to_yaml_string(self_dict)

    def as_dict(self):
        return to_hicklable(self)

    @classmethod
    def load_from_hierarchy(cls, schema_lst):

        REQ = ['name', 'methods']
        ALLOWED = REQ + ['inputs', 'outputs', 'notes']

        REQ_METHOD = ['name', 'implementations']
        ALLOWED_METHOD = REQ_METHOD + ['inputs', 'outputs', 'notes']

        REQ_IMPL = ['name']
        ALLOWED_IMPL = REQ_IMPL + [
            'inputs',
            'outputs',
            'input_map',
            'output_map',
            'commands',
            'command_files',
            'command_pathways',
            'notes',
            'archive_excludes',
        ]

        all_schema_dicts = {}
        for schema in schema_lst:

            name = schema.get('name', 'MISSING NAME')

            bad_keys = set(schema.keys()) - set(ALLOWED)
            miss_keys = set(REQ) - set(schema.keys())

            if bad_keys:
                bad_keys_fmt = ', '.join(['"{}"'.format(i) for i in bad_keys])
                msg = (f'Unknown task schema keys for task schema "{name}": '
                       f'{bad_keys_fmt}.')
                raise TaskSchemaError(msg)

            if miss_keys:
                miss_keys_fmt = ', '.join(['"{}"'.format(i) for i in miss_keys])
                msg = (f'Missing task schema keys for task schema "{name}": '
                       f'{miss_keys_fmt}.')
                raise TaskSchemaError(msg)

            for method in schema['methods']:

                bad_keys = set(method.keys()) - set(ALLOWED_METHOD)
                miss_keys = set(REQ_METHOD) - set(method.keys())

                if bad_keys:
                    bad_keys_fmt = ', '.join(['"{}"'.format(i) for i in bad_keys])
                    msg = (f'Unknown task schema method keys for task schema '
                           f'"{name}": {bad_keys_fmt}.')
                    raise TaskSchemaError(msg)

                if miss_keys:
                    miss_keys_fmt = ', '.join(['"{}"'.format(i) for i in miss_keys])
                    msg = (f'Missing task schema method keys for task schema '
                           f'"{name}": {miss_keys_fmt}.')
                    raise TaskSchemaError(msg)

                for imp in method['implementations']:

                    bad_keys = set(imp.keys()) - set(ALLOWED_IMPL)
                    miss_keys = set(REQ_IMPL) - set(imp.keys())

                    if bad_keys:
                        bad_keys_fmt = ', '.join(['"{}"'.format(i) for i in bad_keys])
                        msg = (f'Unknown task schema implementation keys for task schema '
                               f'"{name}" and method "{method["name"]}": {bad_keys_fmt}.')
                        raise TaskSchemaError(msg)

                    if miss_keys:
                        miss_keys_fmt = ', '.join(['"{}"'.format(i) for i in miss_keys])
                        msg = (f'Missing task schema method keys for task schema '
                               f'"{name}" and method "{method["name"]}": '
                               f'{miss_keys_fmt}.')
                        raise TaskSchemaError(msg)

                    software = SoftwareInstance.get_software_safe(imp['name'])
                    key = (name, method['name'], software)
                    if key in all_schema_dicts:
                        msg = (f'Schema with name "{name}", method "{method["name"]}" '
                               f'and implementation "{software}" is multiply defined.')
                        raise ValueError(msg)

                    input_map = imp.get('input_map', [])
                    output_map = imp.get('output_map', [])
                    command_group = {
                        'commands': imp.get('commands', []),
                        'command_files': imp.get('command_files', {}),
                        'command_pathways': imp.get('command_pathways', [])
                    }
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
                            'implementation': software,
                            'inputs': all_inputs,
                            'outputs': all_outputs,
                            'input_map': input_map,
                            'output_map': output_map,
                            'command_group': command_group,
                            'archive_excludes': imp.get('archive_excludes'),
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

    @property
    def inputs_condensed(self):
        'Get inputs list in their string format.'
        out = []
        for i in self.inputs:
            extra = ''
            i_fmt = f'{i["name"]}'
            if i['alias'] != i['name']:
                extra += f'alias={i["alias"]}'
            if i['context']:
                extra += f'context={i["context"]}'
            if i['group'] != 'default':
                extra += f'group={i["group"]}'
            if 'default' in i:
                extra += f'default={i["default"]!r}'
            if extra:
                i_fmt += f'[{extra}]'
            out.append(i_fmt)
        return out

    def _validate_inputs_outputs(self):
        'Basic checks on inputs and outputs.'

        allowed_inp_specifiers = ['group', 'context', 'alias', 'file', 'default']
        req_inp_keys = ['name']
        allowed_inp_keys = req_inp_keys + allowed_inp_specifiers
        allowed_inp_keys_fmt = ', '.join(['"{}"'.format(i) for i in allowed_inp_keys])

        err = (f'Validation failed for task schema "{self.name}" with method '
               f'"{self.method}" and software "{self.implementation}". ')

        # Normalise schema inputs:
        for inp_idx, inp in enumerate(self.inputs):

            inp_defs = {'context': None, 'group': 'default', 'file': False}
            inp = get_specifier_dict(inp, name_key='name', defaults=inp_defs)

            for r in req_inp_keys:
                if r not in inp:
                    msg = f'Task schema input must include key {r}.'
                    raise TaskSchemaError(err + msg)

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
        for in_map_idx, in_map in enumerate(self.input_map):

            req_keys = ['inputs', 'file']
            allowed_keys = set(req_keys + ['save', 'file_initial'])
            miss_keys = list(set(req_keys) - set(in_map.keys()))
            bad_keys = list(set(in_map.keys()) - allowed_keys)

            msg = (f'Input maps must map a list of `inputs` into a `file` (with an '
                   f'optional `save` key).')
            if miss_keys:
                miss_keys_fmt = ', '.join(['"{}"'.format(i) for i in miss_keys])
                raise TaskSchemaError(err + msg + f' Missing keys are: {miss_keys_fmt}.')
            if bad_keys:
                bad_keys_fmt = ', '.join(['"{}"'.format(i) for i in bad_keys])
                raise TaskSchemaError(err + msg + f' Unknown keys are: {bad_keys_fmt}.')

            if not isinstance(in_map['inputs'], list):
                msg = 'Input map `inputs` must be a list.'
                raise TaskSchemaError(err + msg)

        out_map_opt_names = []
        for out_map_idx, out_map in enumerate(self.output_map):

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

            for out_map_file_idx, out_map_file in enumerate(out_map['files']):
                if ('name' not in out_map_file) or ('save' not in out_map_file):
                    msg = (f'Specify keys `name` (str) and `save` (bool) in output map '
                           f'`files` key.')
                    raise TaskSchemaError(err + msg)

            # Normalise and check output map options:
            out_map_opts = out_map.get('options', [])
            if out_map_opts:
                if not isinstance(out_map_opts, list):
                    msg = (
                        f'If specified, output map options should be a list, but the '
                        f'following was specified: {out_map_opts}.'
                    )
                    raise TaskSchemaError(err + msg)
            for out_map_opt_idx, out_map_opt_i in enumerate(out_map_opts):

                opts = get_specifier_dict(out_map_opt_i, name_key='name')
                req_opts_keys = ['name']
                allowed_opts_keys = req_opts_keys + ['default']
                bad_opts_keys = list(set(opts.keys()) - set(allowed_opts_keys))
                miss_opts_keys = list(set(req_opts_keys) - set(opts.keys()))

                if bad_opts_keys:
                    bad_opts_keys_fmt = ', '.join([f'"{i}"' for i in bad_opts_keys])
                    msg = (
                        f'Unknown output map option keys for output map index '
                        f'{out_map_idx} and output map option index {out_map_opt_idx}: '
                        f'{bad_opts_keys_fmt}. Allowed keys are: {allowed_opts_keys}.'
                    )
                    raise TaskSchemaError(err + msg)

                if miss_opts_keys:
                    miss_opts_keys_fmt = ', '.join([f'"{i}"' for i in miss_opts_keys])
                    msg = (
                        f'Missing output map option keys for output map index '
                        f'{out_map_idx} and output map option index {out_map_opt_idx}: '
                        f'{miss_opts_keys_fmt}.'
                    )
                    raise TaskSchemaError(err + msg)

                if opts['name'] in out_map_opt_names:
                    msg = (
                        f'Output map options must be uniquely named across all output '
                        f'maps of a given task schema, but the output map option '
                        f'"{opts["name"]}" is repeated.'
                    )
                    raise TaskSchemaError(err + msg)
                else:
                    out_map_opt_names.append(opts['name'])

                self.output_map[out_map_idx]['options'][out_map_opt_idx] = opts

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
        """Check for any (local) inputs that are specified but not required by this
        schema."""

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

    def validate_inputs(self, inputs):
        """Check a set of input values are consistent with the schema inputs and populate
        any local input defaults.

        Parameters
        ----------
        inputs : dict of (str : list)


        Returns
        -------
        default_values : dict


        """

        missing_inputs = set(self.input_names) - set(inputs)

        default_values = {}
        for miss_in in missing_inputs:
            miss_in_schema = self.get_input_by_name(miss_in)
            if 'default' in miss_in_schema:
                default_values.update({miss_in: miss_in_schema['default']})
            else:
                msg = (f'Task input "{miss_in}" for task "{self.name}" '
                       f'must be specified because no default value is provided by '
                       f'the schema.')
                raise TaskParameterError(msg)

        return default_values

    def validate_output_map_options(self, options):
        """Check a set of options are consistent with the output map options and populate
        any default values.

        Paramaters
        ----------
        options : dict 
            Output map options specified for the task in the profile. The dict keys are
            checked for consistency with the output map options allowed by the schema.

        Returns
        -------
        opts_validated : dict
            Output map options, as originally passed, but with potentially additional
            options that were not specified, but for which defaults are provided by
            the schema.

        """

        # Collect all option names (across all output maps):
        schema_opts = []
        for out_map in self.output_map:
            schema_opts.extend(out_map.get('options', []))

        opts_validated = copy.deepcopy(options)

        for opt in schema_opts:
            if opt['name'] not in opts_validated:
                if 'default' in opt:
                    opts_validated.update({opt['name']: opt['default']})
                else:
                    msg = (f'Output map option "{opt["name"]}" for task "{self.name}" '
                           f'must be specified because no default value is provided by '
                           f'the schema.')
                    raise TaskParameterError(msg)

        bad_opts = list(set(opts_validated) - set([i['name'] for i in schema_opts]))
        if bad_opts:
            bad_opts_fmt = ', '.join([f'"{i}"' for i in bad_opts])
            msg = (f'Output maps for the schema "{self.name}" are not compatible with '
                   f'the following output map options that are specified in the task: '
                   f'{bad_opts_fmt}.')
            raise TaskParameterError(msg)

        return opts_validated

    @property
    def is_func(self):
        return not self.command_group.commands

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


class TaskStatus(enum.Enum):

    pending = 1
    running = 2
    complete = 3


class Task(object):
    """

    Notes
    -----
    As with `Workflow`, this class is "locked down" quite tightly by using `__slots__` and
    properties. This is to help with maintaining integrity of the workflow between
    save/load cycles.

    """

    __slots__ = [
        '_id',
        '_name',
        '_method',
        '_software_instance',
        '_prepare_software_instance',
        '_process_software_instance',
        '_task_idx',
        '_run_options',
        '_prepare_run_options',
        '_process_run_options',
        '_status',
        '_stats',
        '_context',
        '_local_inputs',
        '_output_map_options',
        '_schema',
        '_resource_usage',
        '_base',
        '_sequences',
        '_repeats',
        '_groups',
        '_nest',
        '_merge_priority',
        '_workflow',
        '_elements',
        '_command_pathway_idx',
    ]

    def __init__(self, workflow, name, method, software_instance,
                 prepare_software_instance, process_software_instance, task_idx,
                 run_options=None, prepare_run_options=None, process_run_options=None,
                 status=None, stats=True, context='', local_inputs=None, schema=None,
                 resource_usage=None, base=None, sequences=None, repeats=None,
                 groups=None, nest=None, merge_priority=None, output_map_options=None,
                 command_pathway_idx=None):

        self._id = None         # Generated once by generate_id()
        self._elements = None   # Assigned in init_elements()

        self._workflow = workflow
        self._name = name
        self._method = method
        self._software_instance = software_instance
        self._prepare_software_instance = prepare_software_instance
        self._process_software_instance = process_software_instance
        self._task_idx = task_idx
        self._run_options = run_options or {}
        self._prepare_run_options = prepare_run_options or {}
        self._process_run_options = process_run_options or {}
        self._status = status or TaskStatus.pending
        self._stats = stats
        self._context = context
        self._local_inputs = local_inputs
        self._output_map_options = output_map_options
        self._schema = schema
        self._resource_usage = resource_usage
        self._command_pathway_idx = command_pathway_idx

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
            f'method={self.method!r}, '
            f'software={self.software!r}'
            f')'
        )
        return out

    def __str__(self):
        return (
            f'{"ID:":10}{self.id!s}\n'
            f'{"Index:":10}{self.task_idx!s}\n'
            f'{"Status:":10}{self.status!s}\n'
            f'{"Name:":10}{self.name!s}\n'
            f'{"Method:":10}{self.method!s}\n'
            f'{"Software:":10}{self.software!s}\n'
            f'{"Context:":10}{(self.context or "DEFAULT")!s}\n'
        )

    def __len__(self):
        return self.local_inputs['length']

    def init_elements(self, elements):
        self._elements = [Element(self, **i) for i in elements]

    def as_dict(self):
        'Return attributes dict with preceding underscores removed.'
        self_dict = {k.lstrip('_'): getattr(self, k) for k in self.__slots__}
        self_dict.pop('workflow')
        self_dict['status'] = (self.status.name, self.status.value)
        self_dict['elements'] = [i.as_dict() for i in self_dict['elements']]
        for i in [
            'software_instance',
            'prepare_software_instance',
            'process_software_instance',
        ]:
            self_dict[i] = self_dict[i].as_dict()

        return self_dict

    def generate_id(self):
        self.id = secrets.token_hex(10)

    @property
    def workflow(self):
        return self._workflow

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
    def name_friendly(self):
        'Capitalise and remove underscores'
        name = '{}{}'.format(self.name[0].upper(), self.name[1:]).replace('_', ' ')
        return name

    @property
    def method(self):
        return self._method

    @property
    def elements(self):
        return tuple(self._elements)

    @property
    def software_instance(self):
        return self._software_instance

    @property
    def prepare_software_instance(self):
        return self._prepare_software_instance

    @property
    def process_software_instance(self):
        return self._process_software_instance

    @property
    def task_idx(self):
        return self._task_idx

    @property
    def run_options(self):
        return {**self.software_instance.required_scheduler_options, **self._run_options}

    @property
    def prepare_run_options(self):
        return {
            **self.prepare_software_instance.required_scheduler_options,
            **self._prepare_run_options,
        }

    @property
    def process_run_options(self):
        return {
            **self.process_software_instance.required_scheduler_options,
            **self._process_run_options,
        }

    def get_scheduler_options(self, task_type='main'):
        """
        Parameters
        ----------
        task_type : str
            One of "main", "prepare", "process"
        """
        run_opts = {
            'main': self.run_options,
            'prepare': self.prepare_run_options,
            'process': self.process_run_options,
        }[task_type]

        non_scheduler_opts = ['num_cores', 'job_array', 'alternate_scratch']
        scheduler_opts = {}
        for k, v in run_opts.items():
            if k in non_scheduler_opts:
                continue
            if k == 'pe':
                v = v + ' ' + str(run_opts['num_cores'])
            scheduler_opts.update({k: v})
        return scheduler_opts

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, status):
        if status not in TaskStatus:
            raise TypeError('`status` must be a `TaskStatus`.')
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
    def output_map_options(self):
        return self._output_map_options

    @property
    def schema(self):
        return self._schema

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
    def software(self):
        return self.software_instance.software

    @property
    def command_pathway_idx(self):
        return self._command_pathway_idx

    def get_formatted_commands(self):
        fmt_commands, input_vars = self.schema.command_group.get_formatted_commands(
            self.local_inputs['inputs'].keys(),
            num_cores=self.run_options['num_cores'],
            cmd_pathway_idx=self.command_pathway_idx,
        )

        # TODO: ?
        # fmt_commands_new = []
        # for i in fmt_commands:
        #     i['line'] = i['line'].replace('<<executable>>', executable)
        #     fmt_commands_new.append(i)
        # fmt_commands = fmt_commands_new

        return fmt_commands, input_vars

    def get_prepare_task_commands(self, is_array=False):
        cmd = f'matflow prepare-task --task-idx={self.task_idx}'
        cmd += f' --array' if is_array else ''
        cmds = [cmd]
        if self.software_instance.task_preparation:
            env_list = self.software_instance.task_preparation.env.as_list()
            cmds = env_list + cmds
        out = [{'subshell': '\n'.join(cmds)}]
        return out

    def get_prepare_task_element_commands(self, is_array=False):
        cmd = (f'matflow prepare-task-element --task-idx={self.task_idx} '
               f'--element-idx=$(($SGE_TASK_ID-1)) '
               f'--directory={self.workflow.path}')
        cmd += f' --array' if is_array else ''
        cmds = [cmd]
        if self.software_instance.task_preparation:
            env_list = self.software_instance.task_preparation.env.as_list()
            cmds = env_list + cmds
        out = [{'subshell': '\n'.join(cmds)}]
        return out

    def get_process_task_commands(self, is_array=False):
        cmd = f'matflow process-task --task-idx={self.task_idx}'
        cmd += f' --array' if is_array else ''
        cmds = [cmd]
        if self.software_instance.task_processing:
            env_list = self.software_instance.task_processing.env.as_list()
            cmds = env_list + cmds
        out = [{'subshell': '\n'.join(cmds)}]
        return out

    def get_process_task_element_commands(self, is_array=False):
        cmd = (f'matflow process-task-element --task-idx={self.task_idx} '
               f'--element-idx=$(($SGE_TASK_ID-1)) '
               f'--directory={self.workflow.path}')
        cmd += f' --array' if is_array else ''
        cmds = [cmd]
        if self.software_instance.task_processing:
            env_list = self.software_instance.task_processing.env.as_list()
            cmds = env_list + cmds
        out = [{'subshell': '\n'.join(cmds)}]
        return out

    @property
    def HDF5_path(self):
        'Get the HDF5 path to this task.'
        return self.workflow.HDF5_path + f'/\'tasks\'/data/data_{self.task_idx}'
