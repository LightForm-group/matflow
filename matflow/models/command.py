"""`matflow.models.command.py`

Module containing functionality for executing commands.

"""

from pathlib import Path, PureWindowsPath, PurePosixPath
from subprocess import run, PIPE
from pprint import pprint

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

    def __init__(self, commands):
        """
        Parameters
        ----------
        commands : list of Command objects
        """

        self.commands = [Command(**i) for i in commands]

    def __repr__(self):
        out = f'{self.__class__.__name__}(commands=['
        out += ', '.join([f'{i!r}' for i in self.commands]) + ']'
        out += ')'
        return out

    def __str__(self):
        return dump_to_yaml_string(self.as_dict())

    def as_dict(self):
        return to_hicklable(self)

    def get_formatted_commands(self, inputs_list, num_cores):
        'Format commands into strings with hpcflow variable substitutions where required.'

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
        self.options = options or []
        self.parameters = parameters or []
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr
        self.parallel_mode = parallel_mode

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
