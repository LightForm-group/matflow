"""`matflow.__init__.py`"""

from matflow._version import __version__

# These dicts map task/method/implementations to specific Python functions.
TASK_INPUT_MAP = {}
TASK_OUTPUT_MAP = {}
TASK_FUNC_MAP = {}
COMMAND_LINE_ARG_MAP = {}
TASK_OUTPUT_FILES_MAP = {}
SOFTWARE_VERSIONS = {}
EXTENSIONS = {}
SCHEMA_IS_VALID = {}

from matflow.api import (
    make_workflow,
    submit_workflow,
    load_workflow,
    append_schema_source,
    prepend_schema_source,
)