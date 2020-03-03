class IncompatibleWorkflow(Exception):
    pass


class IncompatibleNesting(IncompatibleWorkflow):
    pass


class MissingMergePriority(IncompatibleNesting):
    pass
