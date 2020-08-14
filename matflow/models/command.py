"""`matflow.models.command.py`

Module containing functionality for executing commands.

"""

import copy
from pathlib import Path, PureWindowsPath, PurePosixPath
from subprocess import run, PIPE
from pprint import pprint

from matflow.errors import CommandError
from matflow.utils import dump_to_yaml_string
from matflow.hicklable import to_hicklable


def list_formatter(lst):
    return ' '.join([f'{i}' for i in lst])


DEFAULT_FORMATTERS = {
    str: lambda x: x,
    int: lambda number: str(number),
    float: lambda number: f'{number:.6f}',
    list: list_formatter,
    set: list_formatter,
    tuple: list_formatter,
}


class CommandGroup(object):
    """Class to represent a group of commands."""

    def __init__(self, commands, command_files=None):
        """
        Parameters
        ----------
        commands : list of Command objects
        command_files : dict, optional

        """

        self.commands = [Command(**i) for i in commands]
        self.command_files = command_files or {}

        self.resolve_command_files()

    def __repr__(self):
        out = f'{self.__class__.__name__}(commands=['
        out += ', '.join([f'{i!r}' for i in self.commands]) + ']'
        out += ')'
        return out

    def __str__(self):
        return dump_to_yaml_string(self.as_dict())

    def as_dict(self):
        return to_hicklable(self)

    def resolve_command_files(self):

        # Validate command_files dict first:
        for cmd_fn_label, cmd_fn in self.command_files.items():
            if not isinstance(cmd_fn, str) or '<<inc>>' not in cmd_fn:
                msg = ('`command_files` must be a dict that maps a command file label to '
                       'a file name template that must include the substring "<<inc>>", '
                       'which is substituted by increasing integers.')
                raise CommandError(msg)

        file_names = self.get_command_file_names()

        for cmd_idx, command in enumerate(self.commands):

            for opt_idx, opt in enumerate(command.options):
                for opt_token_idx, opt_token in enumerate(opt):
                    options_files = file_names['all_commands'][cmd_idx]['options']
                    for cmd_fn_label, cmd_fn in options_files.items():
                        if f'<<{cmd_fn_label}>>' in opt_token:
                            new_fmt_opt = opt_token.replace(f'<<{cmd_fn_label}>>', cmd_fn)
                            self.commands[cmd_idx].options[opt_idx][opt_token_idx] = new_fmt_opt

            for param_idx, param in enumerate(command.parameters):
                params_files = file_names['all_commands'][cmd_idx]['parameters']
                for cmd_fn_label, cmd_fn in params_files.items():
                    if f'<<{cmd_fn_label}>>' in param:
                        new_param = param.replace(f'<<{cmd_fn_label}>>', cmd_fn)
                        self.commands[cmd_idx].parameters[param_idx] = new_param

            if command.stdin:
                stdin_files = file_names['all_commands'][cmd_idx]['stdin']
                for cmd_fn_label, cmd_fn in stdin_files.items():
                    if f'<<{cmd_fn_label}>>' in command.stdin:
                        new_stdin = command.stdin.replace(f'<<{cmd_fn_label}>>', cmd_fn)
                        self.commands[cmd_idx].stdin = new_stdin

            if command.stdout:
                new_stdout = command.stdout
                stdout_files = file_names['all_commands'][cmd_idx]['stdout']
                for cmd_fn_label, cmd_fn in stdout_files.items():
                    if f'<<{cmd_fn_label}>>' in command.stdout:
                        new_stdout = command.stdout.replace(f'<<{cmd_fn_label}>>', cmd_fn)
                        self.commands[cmd_idx].stdout = new_stdout

            if command.stderr:
                stderr_files = file_names['all_commands'][cmd_idx]['stderr']
                for cmd_fn_label, cmd_fn in stderr_files.items():
                    if f'<<{cmd_fn_label}>>' in command.stderr:
                        new_stderr = command.stderr.replace(f'<<{cmd_fn_label}>>', cmd_fn)
                        self.commands[cmd_idx].stderr = new_stderr

    def get_command_file_names(self):

        out = {
            'input_map': {},
            'output_map': {},
            'all_commands': [],
        }

        file_name_increments = {k: 0 for k in self.command_files.keys()}

        # Input map should use the first increment:
        for cmd_fn_label in self.command_files.keys():
            new_fn = self.command_files[cmd_fn_label].replace(
                '<<inc>>',
                str(file_name_increments[cmd_fn_label]),
            )
            out['input_map'].update({cmd_fn_label: new_fn})

        for command in self.commands:

            file_names_i = {
                'stdin': {},
                'options': {},
                'parameters': {},
                'stdout': {},
                'stderr': {},
            }

            cmd_fn_is_incremented = {k: False for k in self.command_files.keys()}
            for cmd_fn_label in self.command_files.keys():

                for opt in command.options_raw:
                    fmt_opt = list(opt)
                    for opt_token in fmt_opt:
                        if f'<<{cmd_fn_label}>>' in opt_token:
                            new_fn = self.command_files[cmd_fn_label].replace(
                                '<<inc>>',
                                str(file_name_increments[cmd_fn_label]),
                            )
                            file_names_i['stdin'].update({cmd_fn_label: new_fn})

                for param in command.parameters_raw:
                    if f'<<{cmd_fn_label}>>' in param:
                        new_fn = self.command_files[cmd_fn_label].replace(
                            '<<inc>>',
                            str(file_name_increments[cmd_fn_label]),
                        )
                        file_names_i['parameters'].update({cmd_fn_label: new_fn})

                if command.stdin_raw:
                    if f'<<{cmd_fn_label}>>' in command.stdin_raw:
                        new_fn = self.command_files[cmd_fn_label].replace(
                            '<<inc>>',
                            str(file_name_increments[cmd_fn_label]),
                        )
                        file_names_i['stdin'].update({cmd_fn_label: new_fn})

                if command.stdout_raw:
                    if f'<<{cmd_fn_label}>>' in command.stdout_raw:
                        file_name_increments[cmd_fn_label] += 1
                        cmd_fn_is_incremented[cmd_fn_label] = True
                        new_fn = self.command_files[cmd_fn_label].replace(
                            '<<inc>>',
                            str(file_name_increments[cmd_fn_label]),
                        )
                        file_names_i['stdout'].update({cmd_fn_label: new_fn})

                if command.stderr_raw:
                    if f'<<{cmd_fn_label}>>' in command.stderr_raw:
                        if not cmd_fn_is_incremented[cmd_fn_label]:
                            file_name_increments[cmd_fn_label] += 1
                        new_fn = self.command_files[cmd_fn_label].replace(
                            '<<inc>>',
                            str(file_name_increments[cmd_fn_label]),
                        )

                        if not cmd_fn_is_incremented[cmd_fn_label]:
                            cmd_fn_is_incremented[cmd_fn_label] = True
                            file_names_i['stderr'].update({cmd_fn_label: new_fn})

            out['all_commands'].append(file_names_i)

        # Output map should use the final increment:
        for cmd_fn_label in self.command_files.keys():
            new_fn = self.command_files[cmd_fn_label].replace(
                '<<inc>>',
                str(file_name_increments[cmd_fn_label]),
            )
            out['output_map'].update({cmd_fn_label: new_fn})

        return out

    def get_formatted_commands(self, inputs_list, num_cores):
        """Format commands into strings with hpcflow variable substitutions where
        required.

        Parameters
        ----------
        inputs_list : list of str
            List of input names from which a subset of hpcflow variables may be defined.
        num_cores : int
            Number of CPU cores to use for this task. This is required to determine
            whether a "parallel_mode" should be included in the formatted commands.

        Returns
        -------
        tuple of (fmt_commands, var_names)
            fmt_commands : list of dict
                Each list item is a dict that contains keys corresponding to an individual
                command to be run.
            var_names : dict of (str, str)
                A dict that maps a parameter name to an hpcflow variable name.

        """

        fmt_commands = []

        var_names = {}
        for command in self.commands:

            fmt_opts = []
            for opt in command.options:
                fmt_opt = list(opt)
                for opt_token_idx, opt_token in enumerate(fmt_opt):
                    if opt_token in inputs_list:
                        # Replace with an `hpcflow` variable:
                        var_name = 'matflow_input_{}'.format(opt_token)
                        fmt_opt[opt_token_idx] = '<<{}>>'.format(var_name)
                        if opt_token not in var_names:
                            var_names.update({opt_token: var_name})

                fmt_opt_joined = ' '.join(fmt_opt)
                fmt_opts.append(fmt_opt_joined)

            fmt_params = []
            for param in command.parameters:

                fmt_param = param
                if param in inputs_list:
                    # Replace with an `hpcflow` variable:
                    var_name = 'matflow_input_{}'.format(param)
                    fmt_param = '<<{}>>'.format(var_name)

                    if param not in var_names:
                        var_names.update({param: var_name})

                fmt_params.append(fmt_param)

            cmd_fmt = ' '.join([command.command] + fmt_opts + fmt_params)

            if command.stdin:
                cmd_fmt += ' < {}'.format(command.stdin)

            if command.stdout:
                cmd_fmt += ' >> {}'.format(command.stdout)

            if command.stderr:
                if command.stderr == command.stdout:
                    cmd_fmt += ' 2>&1'
                else:
                    cmd_fmt += ' 2>> {}'.format(command.stderr)

            cmd_dict = {'line': cmd_fmt}
            if command.parallel_mode and num_cores > 1:
                cmd_dict.update({'parallel_mode': command.parallel_mode})

            fmt_commands.append(cmd_dict)

        return (fmt_commands, var_names)


class Command(object):
    'Class to represent a command to be executed by a shell.'

    def __init__(self, command, options=None, parameters=None, stdin=None, stdout=None,
                 stderr=None, parallel_mode=None):

        self.command = command
        self.parallel_mode = parallel_mode

        # Raw versions may include command file name variables:
        self.options_raw = options or []
        self.parameters_raw = parameters or []
        self.stdin_raw = stdin
        self.stdout_raw = stdout
        self.stderr_raw = stderr

        # Non-raw versions modified by the parent CommandGroup to include any resolved
        # command file name:
        self.options = copy.deepcopy(self.options_raw)
        self.parameters = copy.deepcopy(self.parameters_raw)
        self.stdin = self.stdin_raw
        self.stdout = self.stdout_raw
        self.stderr = self.stderr_raw

    def __repr__(self):
        out = f'{self.__class__.__name__}({self.command!r}'
        if self.options:
            out += f', options={self.options!r}'
        if self.parameters:
            out += f', parameters={self.parameters!r}'
        if self.stdin:
            out += f', stdin={self.stdin!r}'
        if self.stdout:
            out += f', stdout={self.stdout!r}'
        if self.stderr:
            out += f', stderr={self.stderr!r}'
        out += ')'
        return out

    def __str__(self):

        cmd_fmt = ' '.join(
            [self.command] +
            [' '.join(i) for i in self.options] +
            self.parameters
        )

        if self.stdin:
            cmd_fmt += ' < {}'.format(self.stdin)
        if self.stdout:
            cmd_fmt += ' > {}'.format(self.stdout)
        if self.stderr:
            if self.stderr == self.stdout:
                cmd_fmt += ' 2>&1'
            else:
                cmd_fmt += ' 2> {}'.format(self.stderr)

        return cmd_fmt
