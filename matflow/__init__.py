"""`matflow.__init__.py`"""

import os
import warnings
import pkg_resources
import functools
from ruamel import yaml
from pathlib import Path
import shutil

from matflow._version import __version__
from matflow.errors import MatflowExtensionError
from matflow.validate import validate_task_schemas
from matflow.models.task import TaskSchema

DATA_DIR = Path(os.getenv('MATFLOW_DATA_DIR', '~/.matflow')).expanduser()
DATA_DIR.mkdir(exist_ok=True)

CONFIG_PATH = DATA_DIR.joinpath('config.yml')

if not CONFIG_PATH.is_file():
    # If no config file in data directory, write the default config file:
    def_config = {'task_schema_sources': [str(DATA_DIR.joinpath('task_schemas.yml'))]}
    with CONFIG_PATH.open('w') as handle:
        yaml.safe_dump(def_config, handle)
    # If no task schema file in default location, make one:
    def_schemas = {'software': {}, 'task_schemas': []}
    with DATA_DIR.joinpath('task_schemas.yml').open('w') as handle:
        yaml.safe_dump(def_schemas, handle)

with CONFIG_PATH.open('r') as handle:
    CONFIG = yaml.safe_load(handle)

# Load task_schemas list and software list from all specified task schema files:
_TASK_SCHEMAS = {}
_SOFTWARE = {}
for task_schema_file in CONFIG['task_schema_sources'][::-1]:
    with Path(task_schema_file).open() as handle:
        file_dat = yaml.safe_load(handle)
        task_schemas = file_dat.get('task_schemas', [])
        software = file_dat.get('software', {})
    for i in task_schemas:
        if 'name' not in i:
            raise ValueError('Task schema definition is missing a "name" key.')
        # Overwrite any task schema with the same name (hence we order files in reverse,
        # so e.g. the first task schema file takes precedence):
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

# These dicts map task/method/implementations to specific Python functions.
TASK_INPUT_MAP = {}
TASK_OUTPUT_MAP = {}
TASK_FUNC_MAP = {}
COMMAND_LINE_ARG_MAP = {}
TASK_OUTPUT_FILES_MAP = {}
SOFTWARE_VERSIONS = {}


def input_mapper(input_file, task, method, software):
    """Function decorator for adding input maps from extensions."""
    def _input_mapper(func):
        @functools.wraps(func)
        def func_wrap(*args, **kwargs):
            return func(*args, **kwargs)
        key = (task, method, software)
        if key not in TASK_INPUT_MAP:
            TASK_INPUT_MAP.update({key: {}})
        if input_file in TASK_INPUT_MAP[key]:
            msg = (f'Input file name "{input_file}" already exists in the input map.')
            raise MatflowExtensionError(msg)
        TASK_INPUT_MAP[key].update({input_file: func_wrap})
        return func_wrap
    return _input_mapper


def output_mapper(output_name, task, method, software):
    """Function decorator for adding output maps from extensions."""
    def _output_mapper(func):
        @functools.wraps(func)
        def func_wrap(*args, **kwargs):
            return func(*args, **kwargs)
        key = (task, method, software)
        if key not in TASK_OUTPUT_MAP:
            TASK_OUTPUT_MAP.update({key: {}})
        if output_name in TASK_OUTPUT_MAP[key]:
            msg = (f'Output name "{output_name}" already exists in the output map.')
            raise MatflowExtensionError(msg)
        TASK_OUTPUT_MAP[key].update({output_name: func_wrap})
        return func_wrap
    return _output_mapper


def func_mapper(task, method, software):
    """Function decorator for adding function maps from extensions."""
    def _func_mapper(func):
        @functools.wraps(func)
        def func_wrap(*args, **kwargs):
            return func(*args, **kwargs)
        key = (task, method, software)
        if key in TASK_FUNC_MAP:
            msg = (f'Function map "{key}" already exists in the function map.')
            raise MatflowExtensionError(msg)
        TASK_FUNC_MAP.update({key: func_wrap})
        return func_wrap
    return _func_mapper


def cli_format_mapper(input_name, task, method, software):
    """Function decorator for adding CLI arg formatter functions from extensions."""
    def _cli_format_mapper(func):
        @functools.wraps(func)
        def func_wrap(*args, **kwargs):
            return func(*args, **kwargs)
        key = (task, method, software)
        if key not in COMMAND_LINE_ARG_MAP:
            COMMAND_LINE_ARG_MAP.update({key: {}})
        if input_name in COMMAND_LINE_ARG_MAP[key]:
            msg = (f'Input name "{input_name}" already exists in the CLI formatter map.')
            raise MatflowExtensionError(msg)
        COMMAND_LINE_ARG_MAP[key].update({input_name: func_wrap})
        return func_wrap
    return _cli_format_mapper


def software_versions(software):
    """Function decorator to register an extension function as the function that returns
    a dict of pertinent software versions for that extension."""
    def _software_versions(func):
        @functools.wraps(func)
        def func_wrap(*args, **kwargs):
            return func(*args, **kwargs)
        key = software
        if key in SOFTWARE_VERSIONS:
            msg = (f'Software "{software}" has already registered a `software_versions` '
                   f'function.')
            raise MatflowExtensionError(msg)
        SOFTWARE_VERSIONS[key] = func_wrap
    return _software_versions


def register_output_file(file_reference, file_name, task, method, software):
    key = (task, method, software)
    if key not in TASK_OUTPUT_FILES_MAP:
        TASK_OUTPUT_FILES_MAP.update({key: {}})
    file_ref_full = '__file__' + file_reference
    if file_ref_full in TASK_OUTPUT_FILES_MAP[key]:
        msg = (f'File name "{file_name}" already exists in the output files map.')
        raise MatflowExtensionError(msg)
    TASK_OUTPUT_FILES_MAP[key].update({file_ref_full: file_name})


# From extensions, load functions into the TASK_INPUT_MAP and so on:
EXTENSIONS = {}
extensions_entries = pkg_resources.iter_entry_points('matflow.extension')
if extensions_entries:
    print('Loading extensions...')
    indent = '  '
    for entry_point in extensions_entries:
        loaded = entry_point.load()
        if not hasattr(loaded, '__version__'):
            warnings.warn(f'Matflow extension {entry_point.module_name} has no '
                          f'`__version__` attribute. This extension will not be loaded.')
            continue
        EXTENSIONS.update({
            entry_point.name: {
                'module_name': entry_point.module_name,
                'version': loaded.__version__,
            }
        })
        print(f'{indent}"{entry_point.name}" from {entry_point.module_name} '
              f'(version {loaded.__version__})', flush=True)

    # Validate task schemas against loaded extensions:
    print('Validating task schemas against loaded extensions...', end='')
    try:
        SCHEMA_IS_VALID = validate_task_schemas(
            TASK_SCHEMAS,
            TASK_INPUT_MAP,
            TASK_OUTPUT_MAP,
            TASK_FUNC_MAP
        )
    except Exception as err:
        print('Failed.', flush=True)
        raise err
    num_valid = sum(SCHEMA_IS_VALID.values())
    print(f'OK! {num_valid}/{len(SCHEMA_IS_VALID)} schemas are valid.', flush=True)

else:
    print('No extensions found.')
