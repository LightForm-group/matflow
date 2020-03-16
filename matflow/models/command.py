"""`matflow.models.command.py`

Module containing functionality for executing commands.

"""

from pathlib import Path, PureWindowsPath, PurePosixPath
from subprocess import run, PIPE
from pprint import pprint


class CommandGroup(object):
    """Class to represent a group of commands to be executed within a particular
    environment. Three environment types will eventually be supported: using
    `module load`, activating a `conda` environment, and activating a `venv`
    environment."""

    def __init__(self, commands, env_pre=None, env_post=None):
        """
        Parameters
        ----------
        commands : list of Command objects
        env_pre : list of str
        env_post : list of str
        """

        self.commands = [Command(**i) for i in commands]
        self.env_pre = env_pre or []
        self.env_post = env_post or []

    def __repr__(self):
        out = f'{self.__class__.__name__}(commands=['
        out += ', '.join([f'{i!r}' for i in self.commands]) + ']'
        if self.env_pre:
            env_pre = ', '.join([f'{i!r}' for i in self.env_pre])
            out += ', env_pre=[' + env_pre + ']'
        out += ')'
        return out

    def get_formatted_commands(self, inputs_list):
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
                if param in inputs_list:
                    # Replace with an `hpcflow` variable:
                    var_name = 'matflow_input_{}'.format(param)
                    fmt_param = '<<{}>>'.format(var_name)
                    fmt_params.append(fmt_param)
                    if param not in var_names:
                        var_names.update({param: var_name})

            cmd_fmt = ' '.join([command.command] + fmt_opts + fmt_params)

            if command.stdin:
                cmd_fmt += ' < {}'.format(command.stdin)

            if command.stdout:
                cmd_fmt += ' > {}'.format(command.stdout)

            fmt_commands.append(cmd_fmt)

        return (fmt_commands, var_names)

    def _write_task_executable(self, input_props, os_type, path):

        exec_str = []

        if os_type == 'posix':
            exec_str += ['#!/bin/bash', '']
            ext = 'sh'
            linesep = '\n'
        elif os_type == 'nt':
            ext = 'bat'
            linesep = '\r\n'
        else:
            raise NotImplementedError(f'`os_type` "{os_type}" not supported.')

        if self.env_pre:
            exec_str += self.env_pre + ['']
        exec_str += [i.prepare_execution(input_props) for i in self.commands]
        if self.env_post:
            exec_str += [''] + self.env_post

        task_exec_file = path.joinpath(f'task.{ext}')

        # With an empty string as newline, the line separator is not translated, so we can
        # explicitly set it depending on the target platform (there might be a better way
        # to do this):
        with task_exec_file.open('w', newline='') as handle:
            handle.write(linesep.join(exec_str))

        # Make file executable:
        # if os_type == 'posix':
        #     task_exec_file.chmod(0o666)

        return task_exec_file.name

    def execute_non_scheduled(self, input_props, os_type, path, wsl_wrapper=None):
        """
        Steps:
        1.) form each command with the correct arguments (as a string)
        2.) combine arguments with pre and post environment strings
        3.) write an executable file containing the commands to execute (this
            is like a non-scheduled jobscript I guess...)
        3.) execute file and record stdout and stderr in a new log file
        """

        if wsl_wrapper:
            os_type = 'posix'

        exec_path = self._write_task_executable(input_props, os_type, path)
        run_cmd = f'{exec_path.name}'

        if os_type == 'posix':
            run_cmd = f'./{run_cmd}'

        if wsl_wrapper:
            run_cmd = f'{wsl_wrapper} "{run_cmd}"'

        log_path = path.joinpath('task.log')

        with log_path.open('w') as handle:
            _ = run(run_cmd, shell=True, stdout=handle, stderr=handle,
                    cwd=str(path))

    def prepare_direct_execution(self, input_props, resource, element_path,
                                 wsl_wrapper=None):
        'Prepare command group for non-scheduled execution.'

        # print(f'CommandGroup.prepare_direct.. ')
        # print(f'input_props')
        # pprint(input_props)

        # print(f'resource')
        # pprint(resource)

        # print(f'element_path')
        # pprint(element_path)

        # print(f'wsl_wrapper')
        # pprint(wsl_wrapper)

        try:
            os_type = resource.non_cloud_machines[0]['machine'].os_type
        except ValueError:
            os_type = resource.machine.os_type

        # print(f'ww: {wsl_wrapper}')

        if wsl_wrapper:
            os_type = 'posix'

        # print(f'os_type: {os_type}')

        # run_dir is relative to task dir.
        run_dir = element_path.relative_to(element_path.parent)
        if os_type == 'posix':
            run_dir = PurePosixPath(run_dir)
        elif os_type == 'nt':
            run_dir = PureWindowsPath(run_dir)

        exec_file_name = self._write_task_executable(input_props, os_type, element_path)

        run_cmd = exec_file_name
        if os_type == 'posix':
            run_cmd = f'source {run_cmd}'
        if wsl_wrapper:
            # run_cmd = f'{wsl_wrapper} "/bin/bash -ic \'cd {task_path}; source ~/init_damask.sh; {run_cmd}\'"'
            run_cmd = f'{wsl_wrapper} "{run_cmd}"'

        print(f'prepare_direct_execution: run_cmd: {run_cmd}')

        out = {
            'run_cmd': run_cmd,
            'run_dir': str(run_dir),
        }

        return out

    def prepare_scheduled_execution(self):
        'Prepare command group for scheduled execution.'


class Command(object):
    'Class to represent a command to be executed by a shell.'

    def __init__(self, command, options=None, parameters=None, stdin=None, stdout=None):

        self.command = command
        self.options = options or []
        self.parameters = parameters or []
        self.stdin = stdin
        self.stdout = stdout

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
        out += ')'
        return out
