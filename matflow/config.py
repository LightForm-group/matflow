import os
from pathlib import Path
from warnings import warn

from ruamel.yaml import YAML

from matflow.errors import ConfigurationError
from matflow.models.task import TaskSchema


class Config(object):

    __ALLOWED_CONFIG = ['task_schema_sources']

    __conf = {}

    is_set = False

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

        if Config.is_set:
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
        if not config_file.is_file():
            print('No config.yml found. Generating a config.yml file.')
            def_config = {'task_schema_sources': [str(def_schema_file)]}
            yaml.dump(def_config, config_file)

        if not def_schema_file.is_file():
            def_schemas = {'software': {}, 'task_schemas': []}
            yaml.dump(def_schemas, def_schema_file)

        print(f'Loading matflow config from {config_file}')
        config_dat = yaml.load(config_file)
        bad_keys = list(set(config_dat.keys()) - set(Config.__ALLOWED_CONFIG))
        if bad_keys:
            bad_keys_fmt = ', '.join([f'"{i}"' for i in bad_keys])
            raise ConfigurationError(f'Unknown configuration options: {bad_keys_fmt}.')

        if 'task_schema_sources' not in config_dat:
            msg = (f'Missing `task_schema_sources` from configuration file: '
                   f'{config_file}.')
            raise ConfigurationError(msg)

        return config_dat, config_file

    @staticmethod
    def set_config(config_dir=None):
        'Load configuration from a YAML file.'

        config_dir = Config.resolve_config_dir(config_dir)

        if Config.is_set:
            return

        config_dat, _ = Config.get_config_file(config_dir)
        schema_sources = config_dat['task_schema_sources']

        # Load task_schemas list and software list from all specified task schema files:
        _TASK_SCHEMAS = {}
        _SOFTWARE = {}
        yaml = YAML()
        for task_schema_file in schema_sources[::-1]:

            file_dat = yaml.load(Path(task_schema_file))
            task_schemas = file_dat.get('task_schemas', [])
            software = file_dat.get('software', {})

            for i in task_schemas:
                if 'name' not in i:
                    raise ValueError('Task schema definition is missing a "name" key.')
                # Overwrite any task schema with the same name (hence we order files in
                # reverse so e.g. the first task schema file takes precedence):
                _TASK_SCHEMAS.update({i['name']: i})

            for k, v in software.items():
                _SOFTWARE.update({k: v})

        # Convert to lists:
        _TASK_SCHEMAS = [v for k, v in _TASK_SCHEMAS.items()]
        SOFTWARE = [{**s_dict, 'name': s_name}
                    for s_name, s_list in _SOFTWARE.items()
                    for s_dict in s_list]

        # Load and validate self-consistency of task schemas:
        print('Loading task schemas...', end='')
        try:
            TASK_SCHEMAS = TaskSchema.load_from_hierarchy(_TASK_SCHEMAS)
        except Exception as err:
            print('Failed.')
            raise err
        print('OK!')

        Config.__conf['config_dir'] = config_dir
        Config.__conf['software'] = SOFTWARE
        Config.__conf['task_schemas'] = TASK_SCHEMAS

        Config.is_set = True

    @staticmethod
    def get(name):
        if not Config.is_set:
            raise ConfigurationError('Configuration is not yet set.')
        return Config.__conf[name]
