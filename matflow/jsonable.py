"""`matflow.jsonable.py`"""

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


def to_jsonable(obj, attr=None, parent=None, idx=None, exclude=None):

    if isinstance(obj, list):
        obj_json = []
        for item_idx, item in enumerate(obj):
            obj_json.append(to_jsonable(item, attr, obj, item_idx, exclude))

    elif isinstance(obj, dict):
        obj_json = {}
        for dct_key, dct_val in obj.items():
            obj_json.update(
                {dct_key: to_jsonable(dct_val, attr, obj, dct_key, exclude)})

    elif isinstance(obj, set):
        msg = ('`set` data type is not yet supported by JSONable.')
        raise NotImplementedError(msg)

    elif isinstance(obj, NATIVE_TYPES):
        obj_json = obj

    else:
        # We have an arbitrary object:
        if hasattr(obj, 'to_jsonable'):
            obj_json = obj.to_jsonable()
        else:
            obj_json = {}
            all_attrs = getattr(obj, '__dict__', getattr(obj, '__slots__'))
            for attr, value in all_attrs.items():
                if attr in (exclude or []):
                    continue
                obj_json.update({
                    attr: to_jsonable(value, attr, obj, exclude)
                })

    return obj_json
