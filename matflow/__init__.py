"""`matflow.__init__.py`"""

import os
import pkg_resources
import functools
import yaml
from pathlib import Path
import shutil

from matflow._version import __version__
from matflow.errors import MatflowExtensionError

PKG_DATA_DIR = Path(__file__).parent.joinpath('data')
DATA_DIR = Path(os.getenv('MATFLOW_DATA_DIR', '~/.matflow')).expanduser()
DATA_DIR.mkdir(exist_ok=True)

_CONFIG_PATH = DATA_DIR.joinpath('config.yml')
_TASK_SCHEMAS_FILE_PATH = DATA_DIR.joinpath('task_schemas.yml')

if not _CONFIG_PATH.is_file():
    # If no config file in data directory, copy the default config file:
    shutil.copyfile(
        str(PKG_DATA_DIR.joinpath('config_default.yml')),
        str(_CONFIG_PATH)
    )

with _CONFIG_PATH.open('r') as handle:
    CONFIG = yaml.safe_load(handle)

SOFTWARE = [{**s_dict, 'name': s_name}
            for s_name, s_list in CONFIG['software'].items()
            for s_dict in s_list]

with _TASK_SCHEMAS_FILE_PATH.open() as handle:
    TASK_SCHEMAS = yaml.safe_load(handle)['task_schemas']

# These dicts map task/method/implementations to specific Python functions.
TASK_INPUT_MAP = {}
TASK_OUTPUT_MAP = {}
TASK_FUNC_MAP = {}
COMMAND_LINE_ARG_MAP = {}
TASK_OUTPUT_FILES_MAP = {}


def input_mapper(input_name, task, method, software):
    """Function decorator for adding input maps from extensions."""
    def _input_mapper(func):
        @functools.wraps(func)
        def func_wrap(*args, **kwargs):
            return func(*args, **kwargs)
        key = (task, method, software)
        if key not in TASK_INPUT_MAP:
            TASK_INPUT_MAP.update({key: {}})
        if input_name in TASK_INPUT_MAP[key]:
            msg = (f'Input name "{input_name}" already exists in the input map.')
            raise MatflowExtensionError(msg)
        TASK_INPUT_MAP[key].update({input_name: func_wrap})
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


def func_mapper(task, method):
    """Function decorator for adding function maps from extensions."""
    def _func_mapper(func):
        @functools.wraps(func)
        def func_wrap(*args, **kwargs):
            return func(*args, **kwargs)
        key = (task, method)
        if key in TASK_FUNC_MAP:
            msg = (f'Function map "{key}" already exists in the function map.')
            raise MatflowExtensionError(msg)
        TASK_FUNC_MAP.update({key: func_wrap})
        return func_wrap
    return _func_mapper


def cli_format_mapper(input_name, task, method, software):
    """Function decorator for adding CLI arg formatter functions from extensions."""
    def _func_mapper(func):
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
    return _func_mapper


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
for entry_point in pkg_resources.iter_entry_points('matflow.extension'):
    entry_point.load()
