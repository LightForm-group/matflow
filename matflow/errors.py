class IncompatibleWorkflow(Exception):
    pass


class IncompatibleTaskNesting(IncompatibleWorkflow):
    pass


class MissingMergePriority(IncompatibleTaskNesting):
    pass


class IncompatibleSequence(Exception):
    """For task sequence definitions that are not logically consistent."""


class SequenceError(Exception):
    """For malformed sequence definitions."""


class TaskError(Exception):
    """For malformed task definitions."""


class TaskSchemaError(Exception):
    """For nonsensical task schema definitions."""


class TaskParameterError(Exception):
    """For incorrectly parametrised tasks."""


class ProfileError(Exception):
    """For malformed profile file data."""


class MissingSoftware(Exception):
    """For specified software that cannot be satisfied."""


class WorkflowPersistenceError(Exception):
    """For problems related to saving and loading the persistent HDF5 files."""


class UnsatisfiedGroupParameter(Exception):
    """For when an input has a group, but that group does not exist in the Workflow."""


class MatflowExtensionError(Exception):
    """For problems when loading extensions."""


class MissingSchemaError(Exception):
    """For when a suitable schema does not exist."""


class UnsatisfiedSchemaError(Exception):
    """For when a suitable extension function cannot be found for a task schema."""


class TaskElementExecutionError(Exception):
    """For when the execution of an task element fails."""


class ConfigurationError(Exception):
    """For malformed configuration files."""


class SoftwareInstanceError(Exception):
    """For malformed SoftwareInstance definitions."""
    pass


class MissingSoftwareSourcesError(Exception):
    """For when a software instance requires source variables, but none are forthcoming."""


class UnexpectedSourceMapReturnError(Exception):
    """For when a source map function does not return the expected dict."""


class CommandError(Exception):
    """For problems with command groups and commands."""


class WorkflowIterationError(Exception):
    """For issues with resolving requested iterations."""


class ParameterImportError(Exception):
    """For issues with importing parameters from pre-existing workflows."""
