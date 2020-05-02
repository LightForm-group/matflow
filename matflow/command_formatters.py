"""`matflow.command_formatters.py`

Default formatters for each primitive data type.

"""

DEFAULT_FORMATTERS = {
    str: lambda x: x,
    int: lambda number: str(number),
    float: lambda number: f'{number:.6f}',
    list: list_formatter,
    set: list_formatter,
    tuple: list_formatter,
}


def list_formatter(lst):
    return ' '.join([f'{i}' for i in lst])
