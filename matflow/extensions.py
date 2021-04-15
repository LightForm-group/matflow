import functools
import pkg_resources
import warnings

from matflow.config import Config
from matflow.validation import validate_task_schemas


def load_extensions():

    Config.set_config(raise_on_set=True)
    Config.unlock_extensions()

    extensions_entries = pkg_resources.iter_entry_points('matflow.extension')
    if extensions_entries:
        print('Loading extensions...')
        for entry_point in extensions_entries:

            print(f'  "{entry_point.name}"...', end='', flush=True)

            try:
                loaded = entry_point.load()
            except (ImportError, SyntaxError) as ex:
                print(f'Failed: {ex!r}', flush=True)
                continue

            unload = False

            if not hasattr(loaded, 'SOFTWARE'):
                print('Failed.', flush=True)
                warnings.warn(f'Matflow extension "{entry_point.module_name}" has no '
                              f'`SOFTWARE` attribute. This extension will not be loaded.')
                unload = True

            if not hasattr(loaded, '__version__'):
                print('Failed.', flush=True)
                warnings.warn(f'Matflow extension "{entry_point.module_name}" has no '
                              f'`__version__` attribute. This extension will not be '
                              f'loaded.')
                unload = True

            software_safe = Config._get_software_safe(loaded.SOFTWARE)

            if (
                not unload and
                Config.get('software_versions').get(software_safe) is None
            ):

                # Every defined SoftwareInstance must have a specified version_info:
                version_defined = True
                soft_instances = Config.get('software').get(software_safe)
                if not soft_instances:
                    version_defined = False
                else:
                    for i in soft_instances:
                        if i.version_info is None:
                            version_defined = False
                            break

                if not version_defined:
                    print('Failed.', flush=True)
                    msg = (f'Matflow extension "{entry_point.module_name}" does not '
                           f'register a function for getting software versions and one '
                           f'or more of its software instance definitions do not '
                           f'specify `version_info`. This extension will not be loaded.')
                    warnings.warn(msg)
                    unload = True

            if unload:
                Config.unload_extension(software_safe)
                continue

            Config.set_extension_info(
                entry_point.name,
                {'module_name': entry_point.module_name, 'version': loaded.__version__},
            )
            print(f'(software: "{software_safe}") from '
                  f'{entry_point.module_name} (version {loaded.__version__})', flush=True)

        # Validate task schemas against loaded extensions:
        print('Validating task schemas against loaded extensions...', end='')
        try:
            Config.set_schema_validities(
                validate_task_schemas(
                    Config.get('task_schemas'),
                    Config.get('input_maps'),
                    Config.get('output_maps'),
                    Config.get('func_maps'),
                )
            )
        except Exception as err:
            print('Failed.', flush=True)
            raise err

        schema_validity = Config.get('schema_validity')
        schema_invalids = [(k, v[1]) for k, v in schema_validity.items() if not v[0]]
        num_valid = sum([i[0] for i in schema_validity.values()])
        num_total = len(schema_validity)
        print(f'OK! {num_valid}/{num_total} schemas are valid.', flush=True)
        if schema_invalids:
            sch_invalids_fmt = '\n  '.join([f'{i[0]}: {i[1]}' for i in schema_invalids])
            msg = f'The following schemas are invalid:\n  {sch_invalids_fmt}\n'
            print(msg, flush=True)

    else:
        print('No extensions found.')

    Config.lock_extensions()


def input_mapper(input_file, task, method, software):
    """Function decorator for adding input maps from extensions."""
    def _input_mapper(func):
        @functools.wraps(func)
        def func_wrap(*args, **kwargs):
            return func(*args, **kwargs)
        key = (task, method, software)
        Config.set_input_map(key, input_file, func_wrap)
        return func_wrap
    return _input_mapper


def output_mapper(output_name, task, method, software):
    """Function decorator for adding output maps from extensions."""
    def _output_mapper(func):
        @functools.wraps(func)
        def func_wrap(*args, **kwargs):
            return func(*args, **kwargs)
        key = (task, method, software)
        Config.set_output_map(key, output_name, func_wrap)
        return func_wrap
    return _output_mapper


def func_mapper(task, method, software):
    """Function decorator for adding function maps from extensions."""
    def _func_mapper(func):
        @functools.wraps(func)
        def func_wrap(*args, **kwargs):
            return func(*args, **kwargs)
        key = (task, method, software)
        Config.set_func_map(key, func_wrap)
        return func_wrap
    return _func_mapper


def cli_format_mapper(input_name, task, method, software):
    """Function decorator for adding CLI arg formatter functions from extensions."""
    def _cli_format_mapper(func):
        @functools.wraps(func)
        def func_wrap(*args, **kwargs):
            return func(*args, **kwargs)
        key = (task, method, software)
        Config.set_CLI_arg_map(key, input_name, func_wrap)
        return func_wrap
    return _cli_format_mapper


def software_versions(software):
    """Function decorator to register an extension function as the function that returns
    a dict of pertinent software versions for that extension."""
    def _software_versions(func):
        @functools.wraps(func)
        def func_wrap(*args, **kwargs):
            return func(*args, **kwargs)
        Config.set_software_version_func(software, func_wrap)
        return func_wrap
    return _software_versions


def sources_mapper(task, method, software, **sources_dict):
    """Function decorator to register an extension function that generate task source
    files."""
    def _sources_mapper(func):
        @functools.wraps(func)
        def func_wrap(*args, **kwargs):
            return func(*args, **kwargs)
        key = (task, method, software)
        Config.set_source_map(key, func_wrap, **sources_dict)
        return func_wrap
    return _sources_mapper


def register_output_file(file_reference, file_name, task, method, software):
    key = (task, method, software)
    Config.set_output_file_map(key, file_reference, file_name)
