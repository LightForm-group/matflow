import copy
import socket

from matflow.errors import SoftwareInstanceError, MissingSoftwareSourcesError
from matflow.utils import extract_variable_names


class SoftwareInstance(object):

    __slots__ = [
        '_machine',
        '_software',
        '_label',
        '_environment',
        '_cores_min',
        '_cores_max',
        '_cores_step',
        '_executable',
        '_preparation',
        '_options',
        '_scheduler_options',
    ]

    def __init__(self, software, label=None, environment=None, cores_min=1, cores_max=1,
                 cores_step=1, executable=None, preparation=None, options=None,
                 scheduler_options=None):
        """Initialise a SoftwareInstance object.

        Parameters
        ----------
        software : str
            Name of the software. This is the name that will be exposed as the `SOFTWARE`
            attribute of a Matflow extension package.            
        label : str, optional
            Label used to distinguish software instances for the same `software`. For
            example, this could be a version string.
        environment : str, optional
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
        preparation : dict, optional
            Dict containing the following keys:
                environment : str
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
        scheduler_options : dict, optional
            Scheduler options that are required for using this software instance.

        """

        self._machine = None  # Set once by `set_machine`

        self._software = software
        self._label = label
        self._environment = environment
        self._cores_min = cores_min
        self._cores_max = cores_max
        self._cores_step = cores_step
        self._preparation = preparation
        self._executable = executable
        self._options = options or []
        self._scheduler_options = scheduler_options or {}

        self._validate_num_cores()
        self._validate_preparation()

    def _validate_num_cores(self):
        if self.cores_min < 1:
            raise SoftwareInstanceError('`cores_min` must be greater than 0.')
        if self.cores_min > self.cores_max:
            msg = '`cores_max` must be greater than or equal to `cores_min`.'
            raise SoftwareInstanceError(msg)
        if self.cores_step < 1:
            raise SoftwareInstanceError('`cores_step` must be greater than 0.')

    def _validate_preparation(self):
        if self.preparation:
            ALLOWED = ['environment', 'commands']
            bad_keys = set(self.preparation.keys()) - set(ALLOWED)
            if bad_keys:
                bad_keys_fmt = ', '.join([f'"{i}"' for i in bad_keys])
                raise SoftwareInstanceError(
                    f'Unknown keys for `preparation`: {bad_keys_fmt}.')
            if 'commands' not in self.preparation:
                msg = ('If `preparation` is specified, it must be a dict with at least a '
                       '`commands` key (and optionally an `environment` key).')
                raise SoftwareInstanceError(msg)
            if 'environment' not in self.preparation:
                self._preparation['environment'] = None

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'software={self.software!r}, '
            f'label={self.label!r}, '
            f'cores_range={self.cores_range!r}, '
            f'executable={self.executable!r}'
            f')'
        )

    def as_dict(self):
        'Return attributes dict with preceding underscores removed.'
        self_dict = {k.lstrip('_'): getattr(self, k) for k in self.__slots__}
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
                if i not in sources_map:
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
            'scheduler_options',
            'preparation',
        ]
        INST_ALLOWED = INST_REQUIRED + INST_DICT_KEYS + [
            'label',
            'options',
            'environment',
            'executable',
        ]

        all_instances = {}
        for name, definition in software_dict.items():

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
                for key in (set(INST_ALLOWED) - set(INST_DICT_KEYS)):
                    if key in inst:
                        inst_merged.update({key: inst[key]})

                # Merge values of any `INST_DICT_KEYS` individually.
                for key in INST_DICT_KEYS:
                    if key in inst:
                        if key not in inst_merged:
                            inst_merged.update({key: {}})
                        for subkey in inst[key]:
                            inst_merged[key].update({subkey: inst[key][subkey]})

                inst_merged['software'] = name
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
            '<<sources_dir>>' in self.preparation['commands'] or
            '<<sources_dir>>' in self.executable
        ):
            return True
        else:
            return False

    @property
    def source_variables(self):
        if not self.requires_sources:
            return []
        else:
            source_vars = extract_variable_names(
                self.preparation['commands'] + self.executable,
                ['<<', '>>'],
            )
            return list(set(source_vars) - set(['sources_dir']))

    @property
    def software(self):
        return self._software

    @property
    def label(self):
        return self._label

    @property
    def environment(self):
        return self._environment

    @property
    def environment_str(self):
        return self._environment or ''

    @property
    def environment_lines(self):
        return self.environment_str.splitlines()

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
    def preparation(self):
        return self._preparation

    @property
    def executable(self):
        return self._executable

    @property
    def options(self):
        return self._options

    @property
    def scheduler_options(self):
        return self._scheduler_options

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
