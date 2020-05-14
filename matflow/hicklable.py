"""`matflow.hicklable.py`"""

import numpy as np

NATIVE_TYPES = (
    list,
    dict,
    set,
    int,
    float,
    str,
    np.ndarray,
    np.int32,
    np.int64,
    type(None),
)


def to_hicklable(obj, attr=None, parent=None, idx=None, exclude=None):
    'Get an object representation that can be saved to an HDF5 file using `hickle`.'

    if isinstance(obj, list):
        obj_json = []
        for item_idx, item in enumerate(obj):
            obj_json.append(to_hicklable(item, attr, obj, item_idx, exclude))

    elif isinstance(obj, dict):
        obj_json = {}
        for dct_key, dct_val in obj.items():
            obj_json.update(
                {dct_key: to_hicklable(dct_val, attr, obj, dct_key, exclude)})

    elif isinstance(obj, set):
        msg = ('`set` data type is not yet supported by JSONable.')
        raise NotImplementedError(msg)

    elif isinstance(obj, NATIVE_TYPES):
        obj_json = obj

    else:
        # We have an arbitrary object:
        if hasattr(obj, 'to_hicklable'):
            obj_json = obj.to_hicklable()
        else:

            all_attrs = {}
            if hasattr(obj, '__dict__'):
                all_attrs.update(getattr(obj, '__dict__'))
            if hasattr(obj, '__slots__'):
                all_attrs.update({k: getattr(obj, k) for k in getattr(obj, '__slots__')
                                  if k != '__dict__'})
            if not hasattr(obj, '__dict__') and not hasattr(obj, '__slots__'):
                raise ValueError(f'Object not understood: {obj}.')

            obj_json = {}
            for attr, value in all_attrs.items():
                if attr in (exclude or []):
                    continue
                obj_json.update({
                    attr: to_hicklable(value, attr, obj, exclude)
                })

    return obj_json
