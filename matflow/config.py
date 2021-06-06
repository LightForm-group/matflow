import os
from pathlib import Path
from warnings import warn

from ruamel.yaml import YAML, safe_load


from matflow.errors import ConfigurationError, MatflowExtensionError
from matflow.models.task import TaskSchema
from matflow.models.software import SoftwareInstance


class Config(object):

    __ALLOWED_CONFIG = [
        'task_schema_sources',
        'software_sources',
        'default_run_options',
        'default_preparation_run_options',
        'default_processing_run_options',
        'default_iterate_run_options',
        'default_sticky_run_options',
        'default_sticky_preparation_run_options',
        'default_sticky_processing_run_options',
        'default_sticky_iterate_run_options',
        'parallel_modes',
        'archive_locations',
        'default_metadata',
    ]

    __conf = {}

    _is_set = False
    _is_extension_locked = True

    @staticmethod
    def append_schema_source(schema_source_path, config_dir=None):
        yaml = YAML(typ='rt')
        config_dat, config_file = Config.get_config_file(config_dir=config_dir)
        config_dat['task_schema_sources'].append(str(schema_source_path))
        yaml.dump(config_dat, config_file)

    @staticmethod
    def prepend_schema_source(schema_source_path, config_dir=None):
        yaml = YAML(typ='rt')
        config_dat, config_file = Config.get_config_file(config_dir=config_dir)
        config_dat['task_schema_sources'] = (
            str(schema_source_path) + config_dat['task_schema_sources']
        )
        yaml.dump(config_dat, config_file)

    @staticmethod
    def resolve_config_dir(config_dir=None):

        if not config_dir:
            config_dir = Path(os.getenv('MATFLOW_CONFIG_DIR', '~/.matflow')).expanduser()
        else:
            config_dir = Path(config_dir)

        if Config._is_set:
            if config_dir != Config.get('config_dir'):
                warn(f'Config is already set, but `config_dir` changed from '
                     f'"{Config.get("config_dir")}" to "{config_dir}".')

        if not config_dir.is_dir():
            print('Configuration directory does not exist. Generating.')
            config_dir.mkdir()

        return config_dir

    @staticmethod
    def get_config_file(config_dir):

        yaml = YAML()
        config_file = config_dir.joinpath('config.yml')
        def_schema_file = config_dir.joinpath('task_schemas.yml')
        def_software_file = config_dir.joinpath('software.yml')
        if not config_file.is_file():
            print('No config.yml found. Generating a config.yml file.')
            def_config = {
                'task_schema_sources': [str(def_schema_file)],
                'software_sources': [str(def_software_file)],
                'parallel_modes': {
                    'MPI': {'command': 'mpirun -np <<num_cores>>'},
                    'OpenMP': {'env': 'export OMP_NUM_THREADS=<<num_cores>>'},
                }
            }
            yaml.dump(def_config, config_file)

        if not def_schema_file.is_file():
            print('Generating a default task schema file.')
            yaml.dump([], def_schema_file)

        if not def_software_file.is_file():
            print('Generating a default software file.')
            yaml.dump({}, def_software_file)

        print(f'Loading matflow config from {config_file}')
        with config_file.open() as handle:
            config_dat = safe_load(handle)
        bad_keys = list(set(config_dat.keys()) - set(Config.__ALLOWED_CONFIG))
        if bad_keys:
            bad_keys_fmt = ', '.join([f'"{i}"' for i in bad_keys])
            raise ConfigurationError(f'Unknown configuration options: {bad_keys_fmt}.')

        if 'task_schema_sources' not in config_dat:
            msg = (f'Missing `task_schema_sources` from configuration file: '
                   f'{config_file}.')
            raise ConfigurationError(msg)

        if 'software_sources' not in config_dat:
            msg = f'Missing `software_sources` from configuration file: {config_file}'
            raise ConfigurationError(msg)

        return config_dat, config_file

    @staticmethod
    def set_config(config_dir=None, raise_on_set=False, refresh=False):
        """Load configuration from a YAML file."""

        config_dir = Config.resolve_config_dir(config_dir)

        if Config._is_set:
            if raise_on_set:
                raise ConfigurationError('Configuration is already set.')
            elif not refresh:
                return

        config_dat, _ = Config.get_config_file(config_dir)
        schema_sources = [Path(i).expanduser() for i in config_dat['task_schema_sources']]
        software_sources = [Path(i).expanduser() for i in config_dat['software_sources']]

        # Validate parallel_modes:
        ALLOWED_PARA_MODES = ['MPI', 'OpenMP']
        ALLOWED_PARA_MODES_FMT = ', '.join([f'{i!r}' for i in ALLOWED_PARA_MODES])
        ALLOWED_PARA_CONFIGS = ['env', 'command']
        ALLOWED_PARA_CONFIGS_FMT = ', '.join([f'{i!r}' for i in ALLOWED_PARA_CONFIGS])
        para_modes = {}
        for name, mode_config in config_dat.get('parallel_modes', {}).items():
            if name.lower() not in [i.lower() for i in ALLOWED_PARA_MODES]:
                msg = (f'Parallel mode "{name}" not known. Allowed parallel modes are '
                       f'{ALLOWED_PARA_MODES_FMT}.')
                raise ConfigurationError(msg)
            if not mode_config:
                msg = (f'Specify at least one of {ALLOWED_PARA_CONFIGS_FMT} for parallel '
                       f'mode configuration: "{name}".')
                raise ConfigurationError(msg)
            bad_keys = set(mode_config.keys()) - set(ALLOWED_PARA_CONFIGS)
            if bad_keys:
                bad_keys_fmt = ', '.join([f'{i!r}' for i in bad_keys])
                msg = (f'Unknown parallel mode configuration keys: {bad_keys_fmt} for '
                       f'mode "{name}".')
                raise ConfigurationError(msg)

            if 'env' in mode_config:
                # Split into list of lines:
                mode_config['env'] = mode_config['env'].splitlines()

            # Update to be lowercase:
            para_modes.update({name.lower(): mode_config})

        # Load task_schemas list from all specified task schema files:
        task_schema_dicts = {}
        yaml = YAML(typ='safe')
        for task_schema_file in schema_sources[::-1]:
            if not task_schema_file.is_file():
                msg = f'Task schema source is not a file: "{task_schema_file}".'
                raise ConfigurationError(msg)
            for i in yaml.load(task_schema_file):
                if 'name' not in i:
                    raise ValueError('Task schema definition is missing a "name" key.')
                # Overwrite any task schema with the same name (hence we order files in
                # reverse so e.g. the first task schema file takes precedence):
                task_schema_dicts.update({i['name']: i})

        # Convert to lists:
        task_schema_dicts = [v for k, v in task_schema_dicts.items()]

        # Load and validate self-consistency of task schemas:
        print(f'Loading task schemas from {len(schema_sources)} file(s)...', end='')
        try:
            task_schemas = TaskSchema.load_from_hierarchy(task_schema_dicts)
        except Exception as err:
            print('Failed.')
            raise err
        print('OK!')

        print(f'Loading software definitions from {len(software_sources)} '
              f'file(s)...', end='')
        software = {}
        for software_file in software_sources:
            if not software_file.is_file():
                msg = f'Software source is not a file: "{software_file}".'
                raise ConfigurationError(msg)
            try:
                soft_loaded = SoftwareInstance.load_multiple(yaml.load(software_file))
            except Exception as err:
                print(f'\nFailed to load software definitions from: "{software_file}".')
                raise err

            # Combine software instances from multiple software source files:
            for soft_name, instances in soft_loaded.items():
                if soft_name in software:
                    software[soft_name].extend(instances)
                else:
                    software.update({soft_name: instances})
        print('OK!')

        archive_locs = config_dat.get('archive_locations', {})
        for arch_name, arch in archive_locs.items():
            ALLOWED_ARCH_KEYS = ['path', 'cloud_provider']
            if 'path' not in arch:
                msg = f'Missing `path` for archive location "{arch_name}".'
                raise ConfigurationError(msg)
            bad_keys = set(arch.keys()) - set(ALLOWED_ARCH_KEYS)
            if bad_keys:
                bad_keys_fmt = ', '.join([f'{i!r}' for i in bad_keys])
                msg = (f'Unknown archive location keys for archive "{arch_name}": '
                       f'{bad_keys_fmt}')
                raise ConfigurationError(msg)

            ALLOWED_CLOUD_PROVIDERS = ['dropbox']
            cloud_provider = arch.get('cloud_provider')
            if cloud_provider and cloud_provider not in ALLOWED_CLOUD_PROVIDERS:
                msg = (f'Unsupported cloud provider for archive "{arch_name}": '
                       f'"{cloud_provider}". Supported cloud providers are: '
                       f'{ALLOWED_CLOUD_PROVIDERS}.')
                raise ConfigurationError(msg)

        Config.__conf['config_dir'] = config_dir

        for i in [
            'default_run_options',
            'default_preparation_run_options',
            'default_processing_run_options',
            'default_iterate_run_options',
            'default_sticky_run_options',
            'default_sticky_preparation_run_options',
            'default_sticky_processing_run_options',
            'default_sticky_iterate_run_options',
            'default_metadata',
        ]:
            Config.__conf[i] = config_dat.get(i, {})

        hpcflow_config_dir = config_dir.joinpath('.hpcflow')
        Config.__conf['hpcflow_config_dir'] = hpcflow_config_dir
        Config.__conf['software'] = software
        Config.__conf['task_schemas'] = task_schemas
        Config.__conf['parallel_modes'] = para_modes
        Config.__conf['archive_locations'] = archive_locs

        Config.__conf['input_maps'] = {}
        Config.__conf['output_maps'] = {}
        Config.__conf['func_maps'] = {}
        Config.__conf['CLI_arg_maps'] = {}
        Config.__conf['sources_maps'] = {}
        Config.__conf['output_file_maps'] = {}
        Config.__conf['software_versions'] = {}
        Config.__conf['extension_info'] = {}
        Config.__conf['schema_validity'] = {}

        Config._is_set = True

    @staticmethod
    def get(name):
        if not Config._is_set:
            raise ConfigurationError('Configuration is not yet set.')
        return Config.__conf[name]

    @staticmethod
    def lock_extensions():
        Config._is_extension_locked = True

    @staticmethod
    def unlock_extensions():
        Config._is_extension_locked = False

    @staticmethod
    def _get_software_safe(software_name):
        return SoftwareInstance.get_software_safe(software_name)

    @staticmethod
    def _get_key_safe(key):
        return key[0], key[1], Config._get_software_safe(key[2])

    @staticmethod
    def _validate_extension_setter():
        if not Config._is_set:
            warn(f'Configuration is not yet set. Matflow extension functions will not '
                 'be mapped to task schemas unless matflow is loaded.')
            return False
        if Config._is_extension_locked:
            msg = 'Configuration is locked against modifying extension data.'
            raise ConfigurationError(msg)
        return True

    @staticmethod
    def set_input_map(key, input_file, func):
        if Config._validate_extension_setter():
            key = Config._get_key_safe(key)
            if key not in Config.__conf['input_maps']:
                Config.__conf['input_maps'].update({key: {}})
            if input_file in Config.__conf['input_maps'][key]:
                msg = f'Input file name "{input_file}" already exists in the input map.'
                raise MatflowExtensionError(msg)
            Config.__conf['input_maps'][key][input_file] = func

    @staticmethod
    def set_output_map(key, output_name, func):
        if Config._validate_extension_setter():
            key = Config._get_key_safe(key)
            if key not in Config.__conf['output_maps']:
                Config.__conf['output_maps'].update({key: {}})
            if output_name in Config.__conf['output_maps'][key]:
                msg = f'Output name "{output_name}" already exists in the output map.'
                raise MatflowExtensionError(msg)
            Config.__conf['output_maps'][key][output_name] = func

    @staticmethod
    def set_func_map(key, func):
        if Config._validate_extension_setter():
            key = Config._get_key_safe(key)
            if key in Config.__conf['func_maps']:
                msg = f'Function map "{key}" already exists in the function map.'
                raise MatflowExtensionError(msg)
            Config.__conf['func_maps'][key] = func

    @staticmethod
    def set_CLI_arg_map(key, input_name, func):
        if Config._validate_extension_setter():
            key = Config._get_key_safe(key)
            if key not in Config.__conf['CLI_arg_maps']:
                Config.__conf['CLI_arg_maps'].update({key: {}})
            if input_name in Config.__conf['CLI_arg_maps'][key]:
                msg = (f'Input name "{input_name}" already exists in the CLI formatter '
                       f'map.')
                raise MatflowExtensionError(msg)
            Config.__conf['CLI_arg_maps'][key][input_name] = func

    @staticmethod
    def set_source_map(key, func, **sources_dict):
        if Config._validate_extension_setter():
            key = Config._get_key_safe(key)
            if key in Config.__conf['sources_maps']:
                msg = f'Sources map for key: {key} already exists in.'
                raise MatflowExtensionError(msg)
            Config.__conf['sources_maps'].update({
                key: {'func': func, 'sources': sources_dict}
            })

    @staticmethod
    def set_software_version_func(software, func):
        if Config._validate_extension_setter():
            software = Config._get_software_safe(software)
            if software in Config.__conf['software_versions']:
                msg = (f'Software "{software}" has already registered a '
                       f'`software_versions` function.')
                raise MatflowExtensionError(msg)
            Config.__conf['software_versions'][software] = func

    @staticmethod
    def set_output_file_map(key, file_reference, file_name):
        if Config._validate_extension_setter():
            key = Config._get_key_safe(key)
            if key not in Config.__conf['output_file_maps']:
                Config.__conf['output_file_maps'].update({key: {}})
            file_ref_full = '__file__' + file_reference
            if file_ref_full in Config.__conf['output_file_maps'][key]:
                msg = f'File name "{file_name}" already exists in the output files map.'
                raise MatflowExtensionError(msg)
            Config.__conf['output_file_maps'][key].update({file_ref_full: file_name})

    @staticmethod
    def set_extension_info(name, info):
        if Config._validate_extension_setter():
            if name in Config.__conf['extension_info']:
                msg = f'Extension with name "{name}" already loaded.'
                raise MatflowExtensionError(msg)
            Config.__conf['extension_info'][name] = info

    @staticmethod
    def set_schema_validities(validities):
        if Config._validate_extension_setter():
            Config.__conf['schema_validity'].update(validities)

    @staticmethod
    def unload_extension(software_name):

        name = Config._get_software_safe(software_name)

        in_map = [k for k in Config.__conf['input_maps'] if k[2] == name]
        for k in in_map:
            del Config.__conf['input_maps'][k]

        out_map = [k for k in Config.__conf['output_maps'] if k[2] == name]
        for k in out_map:
            del Config.__conf['output_maps'][k]

        func_map = [k for k in Config.__conf['func_maps'] if k[2] == name]
        for k in func_map:
            del Config.__conf['func_maps'][k]

        CLI_map = [k for k in Config.__conf['CLI_arg_maps'] if k[2] == name]
        for k in CLI_map:
            del Config.__conf['CLI_arg_maps'][k]

        out_file_map = [k for k in Config.__conf['output_file_maps'] if k[2] == name]
        for k in out_file_map:
            del Config.__conf['output_file_maps'][k]

        soft_vers = [k for k in Config.__conf['software_versions'] if k == name]
        for k in soft_vers:
            del Config.__conf['software_versions'][k]

        ext_info = [k for k in Config.__conf['extension_info'] if k == name]
        for k in ext_info:
            del Config.__conf['extension_info'][k]

        schema_valid = [k for k in Config.__conf['schema_validity'] if k[2] == name]
        for k in schema_valid:
            del Config.__conf['schema_validity'][k]

        source_map = [k for k in Config.__conf['sources_maps'] if k[2] == name]
        for k in source_map:
            del Config.__conf['sources_maps'][k]
