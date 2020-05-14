"""`matflow.hicklable.py`"""

import numpy as np

HICKLABLE_PRIMITIVES = (
    int,
    float,
    str,
    np.ndarray,
    np.int32,
    np.int64,
    type(None),
)


def to_hicklable(obj, exclude=None, name_replace=None):
    """Get an object representation that can be saved to an HDF5 file using `hickle`.

    Parameters
    ----------
    obj : object
        Object whose hicklable representation is to be returned.
    exclude : list, optional
        Attributes to exclude from the returned representation
    name_replace : dict
        Strings to find and replace in the object keys.

    """

    if isinstance(obj, (list, tuple, set)):
        obj_valid = []
        for item in obj:
            obj_valid.append(to_hicklable(item, exclude, name_replace))
        if isinstance(obj, tuple):
            obj_valid = tuple(obj_valid)
        elif isinstance(obj, set):
            obj_valid = set(obj_valid)

    elif isinstance(obj, dict):
        obj_valid = {}
        for dct_key, dct_val in obj.items():
            for find, replace in (name_replace or {}).items():
                dct_key = dct_key.replace(find, replace)
            obj_valid.update({dct_key: to_hicklable(dct_val, exclude, name_replace)})

    elif isinstance(obj, HICKLABLE_PRIMITIVES):
        obj_valid = obj

    else:
        # We have an arbitrary object:
        if hasattr(obj, 'to_hicklable'):
            obj_valid = obj.to_hicklable()
        else:

            all_attrs = {}
            if hasattr(obj, '__dict__'):
                all_attrs.update(getattr(obj, '__dict__'))
            if hasattr(obj, '__slots__'):
                all_attrs.update({k: getattr(obj, k) for k in getattr(obj, '__slots__')
                                  if k != '__dict__'})
            if not hasattr(obj, '__dict__') and not hasattr(obj, '__slots__'):
                raise ValueError(f'Object not understood: {obj}.')

            obj_valid = {}
            for attr, value in all_attrs.items():
                if attr in (exclude or []):
                    continue
                for find, replace in (name_replace or {}).items():
                    attr = attr.replace(find, replace)
                obj_valid.update({
                    attr: to_hicklable(value, exclude, name_replace)
                })

    return obj_valid
