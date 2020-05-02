class IncompatibleWorkflow(Exception):
    pass


class IncompatibleTaskNesting(IncompatibleWorkflow):
    pass


class MissingMergePriority(IncompatibleTaskNesting):
    pass


class IncompatibleSequence(Exception):
    'For task sequence definitions that are not logically consistent.'


class TaskSchemaError(Exception):
    'For nonsensical task schema definitions.'


class TaskParameterError(Exception):
    'For incorrectly parametrised tasks.'


class ProfileError(Exception):
    'For malformed profile file data.'


class MissingSoftware(Exception):
    'For specified software that cannot be satisfied.'


class WorkflowPersistenceError(Exception):
    'For problems related to saving and loading the persistent HDF5 files.'
