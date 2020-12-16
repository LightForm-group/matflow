import copy
import socket

from matflow.errors import SoftwareInstanceError, MissingSoftwareSourcesError
from matflow.utils import extract_variable_names


class SourcesPreparation(object):

    __slots__ = ['_commands', '_env']

    def __init__(self, commands=None, env=None):
        self._commands = commands
        self._env = EnvironmentSpec(env)

    def __repr__(self):
        return f'{self.__class__.__name__}(commands={self.commands!r}, env={self.env!r})'

    def __bool__(self):
        return True if self.commands else False

    @property
    def commands(self):
        return self._commands

    def get_formatted_commands(self, source_vars, sources_dir, task_idx):
        out = [{
            'line': (f'matflow prepare-sources '
                     f'--task-idx={task_idx} '
                     f'--iteration-idx=$ITER_IDX')
        }]
        if self.commands:
            for new_cmd in self.commands.splitlines():
                new_cmd = new_cmd.replace('<<sources_dir>>', sources_dir)
                for src_var_name, src_name in source_vars.items():
                    new_cmd = new_cmd.replace(f'<<{src_var_name}>>', src_name)
                out.append({'line': new_cmd})
        return out

    @property
    def commands_fmt(self):
        return [{'line': i} for i in self._commands]

    @property
    def env(self):
        return self._env

    def as_dict(self):
        return {'commands': self.commands, 'env': self.env.value}


class AuxiliaryTaskSpec(object):

    __slots__ = ['_env']

    def __init__(self, env=None):
        self._env = EnvironmentSpec(env)

    def __repr__(self):
        return f'{self.__class__.__name__}(env={self.env!r})'

    @property
    def env(self):
        return self._env

    def as_dict(self):
        return {'env': self.env.value}


class EnvironmentSpec(object):

    __slots__ = ['_value']

    def __init__(self, value=None):
        self._value = value

    def __repr__(self):
        return f'{self.__class__.__name__}(value={self.value!r})'

    @property
    def value(self):
        return self._value

    def as_str(self):
        return self.value or ''

    def as_list(self):
        return self.as_str().splitlines()


class SoftwareInstance(object):

    __slots__ = [
        '_machine',
        '_software_friendly',
        '_label',
        '_env',
        '_cores_min',
        '_cores_max',
        '_cores_step',
        '_executable',
        '_sources_preparation',
        '_options',
        '_required_scheduler_options',
        '_version_info',
        '_task_preparation',
        '_task_processing',
    ]

    def __init__(self, software, label=None, env=None, cores_min=1, cores_max=1,
                 cores_step=1, executable=None, sources_preparation=None, options=None,
                 required_scheduler_options=None, version_info=None,
                 task_preparation=None, task_processing=None):
        """Initialise a SoftwareInstance object.

        Parameters
        ----------
        software : str
            Name of the software. This is the name that will be exposed as the `SOFTWARE`
            attribute of a Matflow extension package.
        label : str, optional
            Label used to distinguish software instances for the same `software`. For
            example, this could be a version string.
        env : str, optional
            Multi-line string containing commands to be executed by the shell that are
            necessary to set up the environment for running this software.
        executable : str, optional
            The command that represents the executable for running this software.
        cores_min : int, optional
            Specifies the minimum number (inclusive) of cores this software instance
            supports. By default, 1.
        cores_max : int, optional
            Specifies the maximum number (inclusive) of cores this software instance
            supports. By default, 1.
        cores_step : int, optional
            Specifies the step size from `cores_min` to `cores_max` this software instance
            supports. By default, 1.
        sources_preparation : dict, optional
            Dict containing the following keys:
                env : str
                    Multi-line string containing commands to be executed by the shell that
                    are necessary to set up the environment for running the preparation
                    commands.
                commands : str
                    Multi-line string containing commands to be executed within the
                    preparation `environment` that are necessary to prepare the
                    executable. For instance, this might contain commands that compile a
                    source code file into an executable.
        options : list of str, optional
            Additional software options as string labels that this instance supports. This
            can be used to label software instances for which add-ons are loaded.
        required_scheduler_options : dict, optional
            Scheduler options that are required for using this software instance.
        version_info : dict, optional
            If an extension does not provide a `software_version` function, then the
            version info dict must be specified here. The keys are str names and the
            values are dicts that must contain at least a key `version`.
        task_preparation : dict, optional
            Dict containing the following keys:
                env : str
                    Multi-line string containing commands to be executed by the shell that
                    are necessary to set up the environment for running
                    `matflow prepare-task`.
        task_processing : dict, optional
            Dict containing the following keys:
                env : str
                    Multi-line string containing commands to be executed by the shell that
                    are necessary to set up the environment for running
                    `matflow process-task`.

        """

        self._machine = None  # Set once by `set_machine`

        self._software_friendly = software
        self._label = label
        self._env = EnvironmentSpec(env)
        self._cores_min = cores_min
        self._cores_max = cores_max
        self._cores_step = cores_step
        self._sources_preparation = SourcesPreparation(**(sources_preparation or {}))
        self._executable = executable
        self._options = options or []
        self._required_scheduler_options = required_scheduler_options or {}
        self._version_info = version_info or None
        self._task_preparation = AuxiliaryTaskSpec(**(task_preparation or {}))
        self._task_processing = AuxiliaryTaskSpec(**(task_processing or {}))

        self._validate_num_cores()
        self._validate_version_infos()

    def _validate_num_cores(self):
        if self.cores_min < 1:
            raise SoftwareInstanceError('`cores_min` must be greater than 0.')
        if self.cores_min > self.cores_max:
            msg = '`cores_max` must be greater than or equal to `cores_min`.'
            raise SoftwareInstanceError(msg)
        if self.cores_step < 1:
            raise SoftwareInstanceError('`cores_step` must be greater than 0.')

    def _validate_version_infos(self):
        if self.version_info:
            REQUIRED = ['version']
            for k, v in self.version_info.items():
                miss_keys = set(REQUIRED) - set(v.keys())
                if miss_keys:
                    miss_keys_fmt = ', '.join([f'"{i}"' for i in miss_keys])
                    msg = (f'Missing required keys in version info dict for name "{k}" '
                           f'for software definition "{self.software}": {miss_keys_fmt}.')
                    raise SoftwareInstanceError(msg)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'software={self.software!r}, '
            f'label={self.label!r}, '
            f'cores_range={self.cores_range!r}, '
            f'executable={self.executable!r}, '
            f'version_info={self.version_info!r}'
            f')'
        )

    def as_dict(self):
        """Return attributes dict with preceding underscores removed."""
        self_dict = {k.lstrip('_'): getattr(self, k) for k in self.__slots__}
        self_dict['software'] = self_dict.pop('software_friendly')
        self_dict['env'] = self_dict['env'].value
        self_dict['sources_preparation'] = self_dict['sources_preparation'].as_dict()
        self_dict['task_preparation'] = self_dict['task_preparation'].as_dict()
        self_dict['task_processing'] = self_dict['task_processing'].as_dict()
        return self_dict

    def validate_source_maps(self, task, method, software, all_sources_maps):
        """Check that any sources required in the preparation commands or executable are
        available in the sources map."""

        source_vars = self.source_variables
        if source_vars:
            if (task, method, software) not in all_sources_maps:
                msg = (f'No extension defines a sources map for the task "{task}" with '
                       f'method "{method}" and software "{software}".')
                raise MissingSoftwareSourcesError(msg)
            else:
                sources_map = all_sources_maps[(task, method, software)]

            for i in source_vars:
                if i not in sources_map['sources']:
                    msg = (f'Source variable name "{i}" is not in the sources map for '
                           f'task "{task}" with method "{method}" and software '
                           f'"{software}".')
                    raise MissingSoftwareSourcesError(msg)

    @classmethod
    def load_multiple(cls, software_dict=None):
        """Load many SoftwareInstance objects from a dict of software instance
        definitions.

        Parameters
        ----------
        software_dict : dict of (str : dict)
            Keys are software names and values are dicts with the following keys:
                instances : list of dict
                    Each element is a dict
                instance_defaults : dict, optional
                    Default values to apply to each dict in the `instances` list.

        Returns
        -------
        all_instances : dict of (str : list of SoftwareInstance)

        """

        software_dict = software_dict or {}
        REQUIRED = ['instances']
        ALLOWED = REQUIRED + ['instance_defaults']

        INST_REQUIRED = ['num_cores']
        INST_DICT_KEYS = [
            'required_scheduler_options',
            'sources_preparation',
        ]
        INST_ALLOWED = INST_REQUIRED + INST_DICT_KEYS + [
            'label',
            'options',
            'env',
            'executable',
            'version_info',
            'task_preparation',
            'task_processing',
        ]

        all_instances = {}
        for name, definition in software_dict.items():

            name_friendly = name
            name = SoftwareInstance.get_software_safe(name)

            bad_keys = set(definition.keys()) - set(ALLOWED)
            miss_keys = set(REQUIRED) - set(definition.keys())
            if bad_keys:
                bad_keys_fmt = ', '.join([f'"{i}"' for i in bad_keys])
                msg = (f'Unknown keys in software instance definitions for software '
                       f'"{name}": {bad_keys_fmt}.')
                raise SoftwareInstanceError(msg)
            if miss_keys:
                miss_keys_fmt = ', '.join([f'"{i}"' for i in miss_keys])
                msg = (f'Software instance definitions for software "{name}" are missing '
                       f'keys: {miss_keys_fmt}.')
                raise SoftwareInstanceError(msg)

            # Merge instance defaults with instance definition:
            inst_defs = definition.get('instance_defaults', {})
            all_name_instances = []
            for inst in definition['instances']:

                inst = dict(inst)
                inst_merged = dict(copy.deepcopy(inst_defs))

                for key, val in inst.items():
                    if key not in INST_DICT_KEYS:
                        inst_merged.update({key: val})

                # Merge values of any `INST_DICT_KEYS` individually.
                for key in INST_DICT_KEYS:
                    if key in inst:
                        if key not in inst_merged:
                            inst_merged.update({key: {}})
                        for subkey in inst[key]:
                            inst_merged[key].update({subkey: inst[key][subkey]})

                bad_keys = set(inst_merged.keys()) - set(INST_ALLOWED)
                miss_keys = set(INST_REQUIRED) - set(inst_merged.keys())

                if bad_keys:
                    bad_keys_fmt = ', '.join([f'"{i}"' for i in bad_keys])
                    msg = (f'Unknown keys in software instance definitions for software '
                           f'"{name}": {bad_keys_fmt}.')
                    raise SoftwareInstanceError(msg)
                if miss_keys:
                    miss_keys_fmt = ', '.join([f'"{i}"' for i in miss_keys])
                    msg = (f'Software instance definitions for software "{name}" are '
                           f'missing keys: {miss_keys_fmt}.')
                    raise SoftwareInstanceError(msg)

                inst_merged['software'] = name_friendly
                num_cores = inst_merged.pop('num_cores', None)
                cores_min = 1
                cores_max = 1
                cores_step = 1
                if num_cores is not None:
                    if isinstance(num_cores, (list, tuple)):
                        if len(num_cores) == 2:
                            cores_min, cores_max = num_cores
                        elif len(num_cores) == 3:
                            cores_min, cores_max, cores_step = num_cores
                        else:
                            msg = (f'`num_cores` value not understood in software '
                                   f'instance definition for software "{name}".')
                            raise SoftwareInstanceError(msg)
                    else:
                        cores_min = num_cores
                        cores_max = num_cores
                        cores_step = num_cores

                inst_merged.update({
                    'cores_min': cores_min,
                    'cores_max': cores_max,
                    'cores_step': cores_step,
                })

                soft_inst = cls(**inst_merged)
                soft_inst.set_machine()
                all_name_instances.append(soft_inst)

            all_instances.update({name: all_name_instances})

        return all_instances

    @property
    def requires_sources(self):
        if (
            (
                self.sources_preparation and
                '<<sources_dir>>' in self.sources_preparation.commands
            ) or
            (self.executable and '<<sources_dir>>' in self.executable)
        ):
            return True
        else:
            return False

    @property
    def source_variables(self):
        if not self.requires_sources:
            return []
        else:
            source_vars = []
            if self.sources_preparation:
                source_vars += extract_variable_names(
                    self.sources_preparation.commands,
                    ['<<', '>>']
                )
            if self.executable:
                source_vars += extract_variable_names(self.executable, ['<<', '>>'])

            return list(set(source_vars) - set(['sources_dir']))

    @property
    def software(self):
        return self.get_software_safe(self.software_friendly)

    @staticmethod
    def get_software_safe(software_name):
        return software_name.lower().replace(' ', '_')

    @property
    def software_friendly(self):
        return self._software_friendly

    @property
    def label(self):
        return self._label

    @property
    def env(self):
        return self._env

    @property
    def task_preparation(self):
        return self._task_preparation

    @property
    def task_processing(self):
        return self._task_processing

    @property
    def cores_min(self):
        return self._cores_min

    @property
    def cores_max(self):
        return self._cores_max

    @property
    def cores_step(self):
        return self._cores_step

    @property
    def cores_range(self):
        return range(self.cores_min, self.cores_max + 1, self.cores_step)

    @property
    def sources_preparation(self):
        return self._sources_preparation

    @property
    def executable(self):
        return self._executable

    @property
    def options(self):
        return self._options

    @property
    def required_scheduler_options(self):
        return self._required_scheduler_options

    @property
    def version_info(self):
        return self._version_info

    @property
    def machine(self):
        return self._machine

    @machine.setter
    def machine(self, machine):
        if self._machine:
            raise ValueError('`machine` is already set.')
        self._machine = machine

    def set_machine(self):
        self.machine = socket.gethostname()
