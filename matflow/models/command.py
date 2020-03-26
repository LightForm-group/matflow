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

        # print(f'prepare_direct_execution: run_cmd: {run_cmd}')

        out = {
            'run_cmd': run_cmd,
            'run_dir': str(run_dir),
        }

        return out

    def prepare_scheduled_execution(self):
        'Prepare command group for scheduled execution.'


class Command(object):
    """Class to represent a command to be executed by a shell.

    Idea is sometimes a task will be submitted via a scheduler, in which case
    the commands associated with the task will be embedded within a jobscript
    file, whereas sometimes a task will be submitted directly to the shell, in
    which case we can use the a series of subprocess.run calls to execute the
    task. Therefore, the relevant abstraction involves a Command object, which
    can be used in both cases.

    Commands for generate_rve with damask are:

    seeds_fromRandom -N {num_grains} -g {res_0} {res_1} {res_2} > orientation.seeds
    geom_fromVoronoi < orientation.seeds > rve.geom

    """

    def __init__(self, cmd, arg_labels=None, flag_labels=None, opts=None,
                 stdin=None, stdout=None, args=None):

        self.cmd = cmd
        self.arg_labels = arg_labels
        self.flag_labels = flag_labels
        self.opts = opts
        self.stdin = stdin
        self.stdout = stdout
        self.args = args

    def set_argument_inputs(self, inputs):
        """Associate inputs to the command, where available."""

        # print(f'\nset_argument_inputs')

        def resolve_input(label, inputs):

            if label.startswith('_file:'):
                val_path = Path(inputs['_files'][label.split('_file:')[1]])
                val = val_path.name
            else:
                val = inputs.get(k)

            return val

        args = None
        flags = None
        opts = None

        if self.arg_labels:
            args = {}
            for k in self.arg_labels.values():
                val = resolve_input(k, inputs)
                args.update({k: val})

        if self.flag_labels:
            flags = {}
            for k in self.flag_labels.values():
                val = inputs.get(k, False)
                flags.update({k: val})

        if self.opts:
            opts = []
            for opt in self.opts:
                val = resolve_input(opt, inputs)

        return args, flags, opts

    def prepare_execution(self, input_props):
        """Generate a string containing the command with arguments."""

        # print(f'prepare_execution.')
        # print(f'input_props: {input_props}')

        cmd_args, cmd_flags, cmd_opts = self.set_argument_inputs(input_props)
        out = f'{self.cmd}'

        # Add arguments to the command:
        if self.arg_labels:

            args = []
            for key, lab in self.arg_labels.items():

                arg_val = cmd_args.get(lab)
                if arg_val is None:
                    msg = ('Input for "{}" is not assigned; cannot execute '
                           'command.'.format(lab))
                    raise ValueError(msg)

                if isinstance(arg_val, list):
                    arg_val = ' '.join([f'{i}' for i in arg_val])

                key_dash = '-' if len(key) == 1 else '--'
                args.append(f'{key_dash}{key} {arg_val}')

            out += ' ' + ' '.join(args)

        if self.flag_labels:

            flags = []
            for key, lab in self.flag_labels.items():

                flag_val = cmd_flags.get(lab)
                if flag_val:
                    key_dash = '-' if len(key) == 1 else '--'
                    flags.append(f'{key_dash}{key}')

            out += ' ' + ' '.join(flags)

        if self.opts:

            out += ' ' + ' '.join(cmd_opts)

        # Add file redirection:
        if self.stdin:
            out += f' < {self.stdin}'

        if self.stdout:
            out += f' > {self.stdout}'

        return out

    def __repr__(self):
        out = f'{self.__class__.__name__}({self.cmd!r}'
        if self.arg_labels:
            out += f', arg_labels={self.arg_labels!r}'
        if self.args:
            out += f', args={self.args!r}'
        if self.stdin:
            out += f', stdin={self.stdin!r}'
        if self.stdout:
            out += f', stdout={self.stdout!r}'
        out += ')'
        return out

    def __str__(self):
        out = f'{self.cmd}'
        if self.arg_labels:
            out += ' ' + ' '.join([f'-{i}' for i in self.arg_labels.keys()])
        if self.stdin:
            out += ' ' + f'< {self.stdin}'
        if self.stdout:
            out += ' ' + f'> {self.stdout}'
        return out
