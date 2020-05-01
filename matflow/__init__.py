"""`matflow.__init__.py`"""

import os
import yaml
from pathlib import Path
import shutil

from matflow._version import __version__

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

# Populate the task input/output maps (this line must be below TASK_INPUT_MAP and so on):
from matflow.software import *
