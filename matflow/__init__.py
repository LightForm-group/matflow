"""`matflow.__init__.py`"""

import os
import yaml
from pathlib import Path
import shutil

from matflow._version import __version__

PKG_DATA_DIR = Path(__file__).parent.joinpath('data')
TEST_WORKFLOWS_DIR = PKG_DATA_DIR.joinpath('test_workflows')
TEST_WORKING_DIR = PKG_DATA_DIR.joinpath('tests_working_dir')
_TASK_SCHEMAS_FILE_PATH = PKG_DATA_DIR.joinpath('task_schemas.yml')

DATA_DIR = Path(os.getenv('MATFLOW_DATA_DIR', '~/.matflow')).expanduser()
DATA_DIR.mkdir(exist_ok=True)

_CONFIG_PATH = DATA_DIR.joinpath('config.yml')
_SOFTWARE_FILE_PATH = DATA_DIR.joinpath('software.yml')

if not _CONFIG_PATH.is_file():
    # If no config file in data directory, copy the default config file:
    shutil.copyfile(
        str(PKG_DATA_DIR.joinpath('config_default.yml')),
        str(_CONFIG_PATH)
    )

with _CONFIG_PATH.open('r') as handle:
    CONFIG = yaml.safe_load(handle)

CURRENT_MACHINE = CONFIG['current_machine']
SOFTWARE = [{**s_dict, 'name': s_name}
            for s_name, s_list in CONFIG['software'].items()
            for s_dict in s_list]

with _TASK_SCHEMAS_FILE_PATH.open() as handle:
    TASK_SCHEMAS = yaml.safe_load(handle)['task_schemas']

# These dicts map task/method/implementations to specific Python functions.
TASK_INPUT_MAP = {}
TASK_OUTPUT_MAP = {}
TASK_FUNC_MAP = {}

# Populate the task input/output maps:
from matflow.software import *

# Check no two resources/machines share the same name:
_all_mach_names = [i['name'] for i in CONFIG['machines']]
if len(_all_mach_names) != len(set(_all_mach_names)):
    msg = f'Multiple machines specified with the same name: {_all_mach_names}.'
    raise ValueError(msg)

_all_res_names = [i['name'] for i in CONFIG['resources']]
if len(_all_res_names) != len(set(_all_res_names)):
    msg = f'Multiple resources specified with the same name: {_all_res_names}.'
    raise ValueError(msg)

# Check each resource has a valid machine:
for i in CONFIG['resources']:
    if i['machine'] not in _all_mach_names:
        msg = f'''Machine "{i['machine']}" of resource "{i['name']}" does not exist.'''
        raise ValueError(msg)

# Check resource connection src/dst resource names:
for i in CONFIG['resource_conns']:

    if i['source'] not in _all_res_names:
        msg = f'''Resource connection source resource "{i['source']}" does not exist.'''
        raise ValueError(msg)

    if i['destination'] not in _all_res_names:
        msg = (f'''Resource connection destination resource "{i['destination']}" does '''
               f'''not exist.''')
        raise ValueError(msg)
