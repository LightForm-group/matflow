
import inspect

from matflow.errors import UnsatisfiedSchemaError


def validate_input_mapper_func(func, task_inputs):
    """Using `inspect`, validate an input mapper callable from a Matflow extension.

    Parameters
    ----------
    func : callable
    task_inputs : list of str
        List of the input name aliases associated with the task schema.

    Notes
    -----
    Checks performed on `func`:
      - check the first argument is named "path"; raise `TypeError` if not;
      - check for one or more additional arguments which are named according to
        a subset of task parameters (passed in `task_inputs`).

    """

    func_params = inspect.signature(func).parameters

    # Check first argument must be "path":
    first_arg = list(func_params.items())[0]
    if first_arg[0] != 'path':
        msg = (f'The first parameter of an input mapper function must be "path" '
               f'but for {func.__name__} is actually "{first_arg[0]}".')
        raise TypeError(msg)
    else:
        # Remove "path" from argument list, for further analysis:
        func_params = dict(func_params)
        del func_params[first_arg[0]]

    bad_params = list(set(func_params) - set(task_inputs))
    if bad_params:
        bad_params_fmt = ', '.join([f'"{i}"' for i in bad_params])
        msg = (f'The following arguments to the input mapper function "{func.__name__}" '
               f'are not known by the schema: {bad_params_fmt}.')
        raise TypeError(msg)


def validate_output_mapper_func(func, num_file_paths, option_names, input_names):
    """Using `inspect`, validate an output mapper callable from a Matflow extension.

    Parameters
    ----------
    func : callable
    num_file_paths : int
        Number of output files specified in the schema's output map.
    option_names : list of str
        List of the names of output map options.
    input_names : list of str
        List of the names of output map inputs.

    Notes
    -----
    Checks performed on `func`:
      - After the first `num_file_paths` arguments, check the remaining arguments names
        coincide exactly with `option_names` + `inputs`.

    """

    func_params = inspect.signature(func).parameters

    # Check num args first
    exp_num_params = num_file_paths + len(option_names) + len(input_names)
    if len(func_params) != exp_num_params:
        msg = (
            f'The output mapper function "{func.__name__}" does not have the expected '
            f'number of arguments: found {len(func_params)} but expected '
            f'{exp_num_params} ({num_file_paths} file path(s) + {len(option_names)} '
            f'options parameters + {len(input_names)} inputs).'
        )
        raise TypeError(msg)

    # Check option names:
    params = list(func_params.items())[num_file_paths:]
    params_func = [i[0] for i in params]

    miss_params = list(set(option_names + input_names) - set(params_func))
    bad_params = list(set(params_func) - set(option_names + input_names))

    if bad_params:
        bad_params_fmt = ', '.join([f'"{i}"' for i in bad_params])
        msg = (f'The following arguments in the output mapper function "{func.__name__}" '
               f'are not output map options or inputs: {bad_params_fmt}.')
        raise TypeError(msg)

    if miss_params:
        miss_params_fmt = ', '.join([f'"{i}"' for i in miss_params])
        msg = (f'The following output mapper options and/or inputs are missing from the '
               f'signature of the output mapper function "{func.__name__}": '
               f'{miss_params_fmt}.')
        raise TypeError(msg)


def validate_func_mapper_func(func, task_inputs):
    """Using `inspect`, validate an input mapper callable from a Matflow extension.

    Parameters
    ----------
    func : callable
    task_inputs : list of str
        List of the input name aliases associated with the task schema.

    Notes
    -----
    Checks performed on `func`:
      - check function arguments are named according to all task parameters (passed in
      `task_inputs`).

    """

    func_params = inspect.signature(func).parameters

    bad_params = list(set(func_params) - set(task_inputs))
    miss_params = list(set(task_inputs) - set(func_params))

    if bad_params:
        bad_params_fmt = ', '.join([f'"{i}"' for i in bad_params])
        msg = (f'The function mapper function "{func.__name__}" contains the following '
               f'arguments that are not consistent with the schema: {bad_params_fmt}.')
        raise TypeError(msg)

    if miss_params:
        miss_params_fmt = ', '.join([f'"{i}"' for i in miss_params])
        msg = (f'The following task inputs are missing from the signature of the '
               f'function mapper function "{func.__name__}": {miss_params_fmt}.')
        raise TypeError(msg)


def validate_task_schemas(task_schemas, task_input_map, task_output_map, task_func_map):
    """
    Determine whether each task schema is valid.

    Parameters
    ----------
    task_schemas : dict of (tuple : TaskSchema)
        Dict keys are (task_name, task_method, software).
    task_input_map : dict of (tuple : dict of (str : callable))
        Outer dict keys are (task_name, task_method, software); inner dicts map a string
        input file name to a MatFlow extension callable which writes that input file.
    task_output_map : dict of (tuple : dict of (str : callable))
        Outer dict keys are (task_name, task_method, software); inner dicts map a string
        output name to a MatFlow extension callable which return that output.
    task_func_map : dict of (tuple : callable)
        Dict keys are (task_name, task_method, software); values are MatFlow extension
        callables.

    Returns
    -------
    schema_is_valid : dict of (tuple : tuple of (bool, str))
        Dict keys are (task_name, task_method, software); values are tuples whose first
        values are boolean values indicating if a given schema is valid. If False, this
        indicates that one of extension functions (input map, output map or function map)
        is missing. Note that this function does not raise any exception in this case ---
        but the task schema will be noted as invalid. The second value of the dict value
        tuple is a string description of the reason why the schema is invalid.

    Raises
    ------
    UnsatisfiedSchemaError
        Raised if any of the extension callables (input/output/func maps) are not
        consistent with their associated task schema.

    """

    schema_is_valid = {}

    for key, schema in task_schemas.items():

        schema_is_valid.update({key: (True, '')})

        key_msg = (f'Unresolved task schema for task "{schema.name}" with method '
                   f'"{schema.method}" and software "{schema.implementation}".')

        for inp_map in schema.input_map:

            extension_inp_maps = task_input_map.get(key)
            msg = (
                f'{key_msg} No matching extension function found for the input '
                f'map that generates the input file "{inp_map["file"]}".'
            )

            if not extension_inp_maps:
                reason = (f'No input map function found for input map that generates file'
                          f' "{inp_map["file"]}". ')
                schema_is_valid[key] = (False, schema_is_valid[key][1] + reason)
                continue
            else:
                inp_map_func = extension_inp_maps.get(inp_map['file'])
                if not inp_map_func:
                    raise UnsatisfiedSchemaError(msg)

            # Validate signature of input map function:
            try:
                validate_input_mapper_func(inp_map_func, inp_map['inputs'])
            except TypeError as err:
                raise UnsatisfiedSchemaError(key_msg + ' ' + str(err)) from None

        for out_map in schema.output_map:

            extension_out_maps = task_output_map.get(key)
            msg = (
                f'{key_msg} No matching extension function found for the output '
                f'map that generates the output "{out_map["output"]}".'
            )

            if not extension_out_maps:
                reason = (f'No output map function found for output map that generates '
                          f'output "{out_map["output"]}". ')
                schema_is_valid[key] = (False, schema_is_valid[key][1] + reason)
                continue
            else:
                out_map_func = extension_out_maps.get(out_map['output'])
                if not out_map_func:
                    raise UnsatisfiedSchemaError(msg)

            # Validate signature of output map function:
            try:
                validate_output_mapper_func(
                    func=out_map_func,
                    num_file_paths=len(out_map['files']),
                    option_names=[i['name'] for i in out_map.get('options', [])],
                    input_names=[i['name'] for i in out_map.get('inputs', [])],
                )
            except TypeError as err:
                raise UnsatisfiedSchemaError(key_msg + ' ' + str(err)) from None

        if schema.is_func:

            func = task_func_map.get(key)
            if not func:
                reason = 'No function mapper function found. '
                schema_is_valid[key] = (False, schema_is_valid[key][1] + reason)
                continue

            # Validate signature of func mapper function:
            try:
                validate_func_mapper_func(func, schema.input_aliases)
            except TypeError as err:
                raise UnsatisfiedSchemaError(key_msg + ' ' + str(err)) from None

    return schema_is_valid
