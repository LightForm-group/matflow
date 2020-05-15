
import inspect

from matflow.errors import UnsatisfiedSchemaError


def validate_function_args(func, func_type, expected_args=None, num_args=None):
    """
    Parameters
    ----------
    func : callable
    func_type : str
        One of "input_mapper", "output_mapper", ...
    expected_args : list, optional
        Must be specified if `func_type` is "input_mapper" or "func_mapper".
    num_args : int, optional
        Must be specified if `func_type` is "output_mapper".

    """

    func_type_fmt = func_type.replace('_', ' ')
    func_params = inspect.signature(func).parameters

    if func_type == 'input_mapper':
        # Check first argument must be "path":
        first_arg = list(func_params.items())[0]
        if first_arg[0] != 'path':
            msg = (f'The first parameter of an {func_type_fmt} function must be "path" '
                   f'but for {func.__name__} is actually "{first_arg[0]}".')
            raise TypeError(msg)
        else:
            func_params = dict(func_params)
            del func_params[first_arg[0]]

    elif func_type == 'output_mapper':
        # Check num args:
        if len(func_params) != num_args:
            msg = (f'There are {len(func_params)} parameters in output mapping function '
                   f'"{func.__name__}", but {num_args} are expected.')
            raise TypeError(msg)

    if func_type in ['input_mapper', 'func_mapper']:
        non_defaulted = [k for k, v in func_params.items() if v.default == inspect._empty]
        # Parameters for which their are no defaults must be supplied by schema:
        miss_params = list(set(non_defaulted) - set(expected_args))
        if miss_params:
            miss_params_fmt = ', '.join([f'"{i}"' for i in miss_params])
            msg = (f'The following parameters in {func_type_fmt} function '
                   f'"{func.__name__}" have no default value and are not specified in '
                   f'the schema: {miss_params_fmt}.')
            raise TypeError(msg)

        # All parameters in the schema must appear in the function:
        bad_params = list(set(expected_args) - set(func_params))
        if bad_params:
            bad_params_fmt = ', '.join([f'"{i}"' for i in bad_params])
            msg = (f'The following schema input parameters are not compatible with the '
                   f'{func_type_fmt} function "{func.__name__}": {bad_params_fmt}.')
            raise TypeError(msg)


def validate_task_schemas(task_schemas, task_input_map, task_output_map, task_func_map):

    schema_is_valid = {}

    for key, schema in task_schemas.items():

        schema_is_valid.update({key: True})

        key_msg = (f'Unresolved task schema for task "{schema.name}" with method '
                   f'"{schema.method}" and software "{schema.implementation}".')

        for inp_map in schema.input_map:

            extension_inp_maps = task_input_map.get(key)
            msg = (
                f'{key_msg} No matching extension function found for the input '
                f'map that generates the input file "{inp_map["file"]}".'
            )

            if not extension_inp_maps:
                schema_is_valid.update({key: False})
                continue
            else:
                inp_map_func = extension_inp_maps.get(inp_map['file'])
                if not inp_map_func:
                    raise UnsatisfiedSchemaError(msg)

            # Validate input map inputs against func args:
            try:
                validate_function_args(
                    func=inp_map_func,
                    func_type='input_mapper',
                    expected_args=inp_map['inputs'],
                )
            except TypeError as err:
                raise UnsatisfiedSchemaError(key_msg + ' ' + str(err)) from None

        for out_map in schema.output_map:

            extension_out_maps = task_output_map.get(key)
            msg = (
                f'{key_msg} No matching extension function found for the output '
                f'map that generates the output "{out_map["output"]}".'
            )

            if not extension_out_maps:
                schema_is_valid.update({key: False})
                continue
            else:
                out_map_func = extension_out_maps.get(out_map['output'])
                if not out_map_func:
                    raise UnsatisfiedSchemaError(msg)

            # Validate number of output map args
            try:
                validate_function_args(
                    func=out_map_func,
                    func_type='output_mapper',
                    num_args=(len(out_map['files']) + len(out_map.get('options', {}))),
                )
            except TypeError as err:
                raise UnsatisfiedSchemaError(key_msg + ' ' + str(err)) from None

        if schema.is_func:

            func = task_func_map.get(key)
            if not func:
                schema_is_valid.update({key: False})
                continue

            try:
                validate_function_args(
                    func=func,
                    func_type='func_mapper',
                    expected_args=schema.input_aliases,
                )
            except TypeError as err:
                raise UnsatisfiedSchemaError(key_msg + ' ' + str(err)) from None

    return schema_is_valid
