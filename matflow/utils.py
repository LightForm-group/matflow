"""`matflow.utils.py`"""

import sys
import io
import collections
import copy
import itertools
import numpy as np
import random
import re
import time
from contextlib import redirect_stdout

from ruamel.yaml import YAML


def parse_times(format_str):
    """Parse a string which contain time format code and one or
    more `%%r` to represent a random digit from 0 to 9."""

    time_parsed = time.strftime(format_str)
    rnd_all = ''
    while '%r' in time_parsed:
        rnd = str(random.randint(0, 9))
        rnd_all += rnd
        time_parsed = time_parsed.replace('%r', rnd, 1)

    return time_parsed, rnd_all


def zeropad(num, largest):
    """Return a zero-padded string of a number, given the largest number.

    TODO: want to support floating-point numbers as well? Or rename function
    accordingly.

    Parameters
    ----------
    num : int
        The number to be formatted with zeros padding on the left.
    largest : int
        The number that determines the number of zeros to pad with.

    Returns
    -------
    padded : str
        The original number, `num`, formatted as a string with zeros added
        on the left.

    """

    num_digits = len('{:.0f}'.format(largest))
    padded = '{0:0{width}}'.format(num, width=num_digits)

    return padded


def combine_list_of_dicts(a):

    a = copy.deepcopy(a)

    for i in range(1, len(a)):
        update_dict(a[0], a[i])

    return a[0]


def update_dict(base, upd):
    """Update an arbitrarily-nested dict."""

    for key, val in upd.items():
        if isinstance(base, collections.Mapping):
            if isinstance(val, collections.Mapping):
                r = update_dict(base.get(key, {}), val)
                base[key] = r
            else:
                base[key] = upd[key]
        else:
            base = {key: upd[key]}

    return base


def nest_lists(my_list):
    """
        `a` is a list of `N` sublists.

        E.g.
        my_list = [
            [1,2],
            [3,4,5],
            [6,7]
        ]

        returns a list of lists of length `N` such that all combinations of elements from sublists in
        `a` are found
        E.g
        out = [
            [1, 3, 6],
            [1, 3, 7],
            [1, 4, 6],
            [1, 4, 7],
            [1, 5, 6],
            [1, 5, 7],
            [2, 3, 6],
            [2, 3, 7],
            [2, 4, 6],
            [2, 4, 7],
            [2, 5, 6],
            [2, 5, 7]
        ]

    """

    N = len(my_list)
    sub_len = [len(i) for i in my_list]

    products = np.array([1] * (N + 1))
    for i in range(len(my_list) - 1, -1, -1):
        products[:i + 1] *= len(my_list[i])

    out = [[None for x in range(N)] for y in range(products[0])]

    for row_idx, row in enumerate(out):

        for col_idx, col in enumerate(row):

            num_repeats = products[col_idx + 1]
            sub_list_idx = int(row_idx / num_repeats) % len(my_list[col_idx])
            out[row_idx][col_idx] = copy.deepcopy(
                my_list[col_idx][sub_list_idx])

    return out


def repeat(lst, reps):
    'Repeat 1D list elements.'
    return list(itertools.chain.from_iterable(itertools.repeat(x, reps) for x in lst))


def tile(lst, tiles):
    'Tile a 1D list.'
    return lst * tiles


def index(lst, idx):
    'Get elements of a list.'
    return [lst[i] for i in idx]


def arange(size):
    'Get 1D list of increasing integers.'
    return list(range(size))


def extend_index_list(lst, repeats):
    """Extend an integer index list by repeating some number of times such that the extra
    indices added are new and follow the same ordering as the existing elements.

    Parameters
    ----------
    lst : list of int
    repeats : int

    Returns
    -------
    new_idx : list of int
        Returned list has length `len(lst) * repeats`.

    Examples
    --------
    >>> extend_index_list([0, 1, 2], 2)
    [0, 1, 2, 3, 4, 5]

    >>> extend_index_list([0, 0, 1, 1], 3)
    [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]

    >>> extend_index_list([4, 1, 2], 2)
    [4, 1, 2, 8, 5, 6]

    """

    new_idx = []
    for i in lst:
        if i < 0:
            raise ValueError('List elements must be positive or zero.')
        new_idx.append(i)

    for _ in range(repeats - 1):
        next_avail_idx = max(new_idx) + 1
        new_idx.extend([next_avail_idx + i - min(lst) for i in lst])

    return new_idx


def flatten_list(lst):
    """Flatten a list of lists.

    Parameters
    ----------
    lst : list of list

    Returns
    -------
    list 

    Examples
    --------
    >>> flatten_list([[0, 2, 4], [9, 1]])
    [0, 2, 4, 9, 1]

    """
    return [j for i in lst for j in i]


def to_sub_list(lst, sub_list_len):
    """Transform a list into a list of sub lists of certain size.

    Parameters
    ----------
    lst : list
        List to transform into a list of sub-lists.
    sub_list_len : int
        Size of sub-lists. Must be an integer factor of the length of the
        original list, `lst`.

    Returns
    -------
    list of list

    Examples
    --------
    >>> to_sub_list([0, 1, 2, 3], 2)
    [[0, 1], [2, 3]]

    """

    if (sub_list_len <= 0) or (len(lst) % sub_list_len != 0):
        raise ValueError('`sub_list_len` must be a positive factor of `len(lst)`.')
    out = [lst[(i * sub_list_len):((i * sub_list_len) + sub_list_len)]
           for i in range(len(lst) // sub_list_len)]
    return out


def datetime_to_dict(dt):
    return {
        'year': dt.year,
        'month': dt.month,
        'day': dt.day,
        'hour': dt.hour,
        'minute': dt.minute,
        'second': dt.second,
        'microsecond': dt.microsecond,
    }


def dump_to_yaml_string(data):
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    with redirect_stdout(io.StringIO()) as buffer:
        yaml.dump(data, sys.stdout)
        output = buffer.getvalue()
    return output


def get_specifier_dict(key, name_key=None, base_key=None, defaults=None,
                       list_specifiers=None):
    """Resolve a string key with additional specifiers using square-brackets into a dict.

    Parameters
    ----------
    key : str or dict
    name_key : str
    base_key : str
    defaults : dict
    list_specifiers : list of str
        Any specifier in this list will be added to the returned dict as a list element.

    Returns
    -------
    dict

    Examples
    --------
    >>> get_specifier_dict(
        'parameter_1[hey, label_2=hi]',        
        name_key='param_name',
        base_key='label_1',
        defaults={'a': 1},
    )
    {
        'param_name': 'parameter_1',
        'label_1': 'hey'
        'label_2': 'hi',
        'a': 1,
    }

    """

    list_specifiers = list_specifiers or []
    out = {}

    if isinstance(key, str):

        if name_key is None:
            raise TypeError('`name_key` must be specified.')

        match = re.search(r'([\w\-\s]+)(\[(.*?)\])*', key)
        name = match.group(1)
        out.update({name_key: name})

        specifiers_str = match.group(3)
        if specifiers_str:
            base_keys = []
            for s in specifiers_str.split(','):
                if not s:
                    continue
                if '=' in s:
                    s_key, s_val = [i.strip() for i in s.split('=')]
                    if s_key in list_specifiers:
                        if s_key in out:
                            out[s_key].append(s_val)
                        else:
                            out[s_key] = [s_val]
                    else:
                        if s_key in out:
                            raise ValueError(
                                f'Specifier "{s_key}" multiply defined. Add this '
                                f'specifier to `list_specifiers` to add multiple values '
                                f'to the returned dict (in a list).'
                            )
                        out.update({s_key: s_val})
                else:
                    base_keys.append(s.strip())

            if len(base_keys) > 1:
                raise ValueError('Only one specifier may be specified without a key.')

            if base_keys:
                if base_key is None:
                    raise ValueError('Base key found but `base_key` name not specified.')
                out.update({base_key: base_keys[0]})

    elif isinstance(key, dict):
        out.update(key)

    else:
        raise TypeError('`key` must be a dict or str to allow specifiers to be resolved.')

    for k, v in (defaults or {}).items():
        if k not in out:
            out[k] = copy.deepcopy(v)

    return out


def extract_variable_names(source_str, delimiters):
    """Given a specified syntax for embedding variable names within a string,
    extract all variable names.

    Parameters
    ----------
    source_str : str
        The string within which to search for variable names.
    delimiters : two-tuple of str
        The left and right delimiters of a variable name.

    Returns
    -------
    var_names : list of str
        The variable names embedded in the original string.   

    """

    delim_esc = [re.escape(i) for i in delimiters]
    pattern = delim_esc[0] + r'(.\S+?)' + delim_esc[1]
    var_names = re.findall(pattern, source_str)

    return var_names
