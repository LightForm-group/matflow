class IncompatibleWorkflow(Exception):
    pass


class IncompatibleTaskNesting(IncompatibleWorkflow):
    pass


class MissingMergePriority(IncompatibleTaskNesting):
    pass
