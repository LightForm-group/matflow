"""Module containing functionality for generating Python scripts as task sources."""

import re
from textwrap import dedent

import black
import autopep8

from pkg_resources import resource_string


def main_func(func):
    """Decorator used to annotate which function within a snippet is the main function."""
    def main_inner(*args, **kwargs):
        func(*args, **kwargs)
    return main_inner


def get_snippet(package_name, snippet_name, decorator=True):
    """Get a Python snippet function (as a string) from the snippets directory."""
    out = resource_string(package_name, f'snippets/{snippet_name}').decode()
    if not decorator:
        # Remove the `@main_func` decorator and import.
        remove_lns = ['from matflow.scripting import main_func', '@main_func']
        for i in remove_lns:
            out = ''.join(out.split(i))
    return out


def parse_python_func_return(func_str):
    """Get a list of the variable names in a Python function return statement.

    The return statement may return a tuple (with parenthesis or not) or a single variable.

    """

    out = []
    match = re.search(r'return \(*([\S\s][^\)]+)\)*', func_str)
    if match:
        match_clean = match.group(1).strip().strip(',')
        out = [i.strip() for i in match_clean.split(',')]

    return out


def parse_python_func_imports(func_str):
    """Get a list of import statement lines from a (string) Python function."""

    import_lines = func_str.split('def ')[0].strip()
    match = re.search(r'((?:import|from)[\S\s]*)', import_lines)
    out = []
    if match:
        out = match.group(1).splitlines()

    return out


def extract_snippet_main(snippet_str):
    """Extract only the snippet main function (plus imports), as annotated by the
    `@mainfunc` decorator."""

    func_start_pat = r'((?:@main_func\n)?def\s(?:.*)\((?:[\s\S]*?)\):)'

    func_split_snip = re.split(func_start_pat, snippet_str)
    imports = func_split_snip[0]
    main_func_dec_str = '@main_func'

    main_func_str = None
    for idx in range(1, len(func_split_snip[1:]), 2):
        func_str = func_split_snip[idx] + func_split_snip[idx + 1]
        if main_func_dec_str in func_str:
            if main_func_str:
                msg = (f'`{main_func_dec_str}` should decorate only one function within '
                       f'the snippet.')
                raise ValueError(msg)
            else:
                main_func_str = func_str.lstrip(f'{main_func_dec_str}\n')

    imports = ''.join(imports.split('from matflow_defdap import main_func'))

    return imports + '\n' + main_func_str


def get_snippet_signature(package_name, script_name):
    """Get imports, inputs and outputs of a Python snippet function."""

    snippet_str = get_snippet(package_name, script_name)
    snippet_str = extract_snippet_main(snippet_str)

    def_line = re.search(r'def\s(.*)\(([\s\S]*?)\):', snippet_str).groups()
    func_name = def_line[0]
    func_ins = [i.strip() for i in def_line[1].split(',')]

    if script_name != func_name + '.py':
        msg = ('For simplicity, the snippet main function name should be the same as the '
               'snippet file name.')
        raise ValueError(msg)

    func_outs = parse_python_func_return(snippet_str)
    func_imports = parse_python_func_imports(snippet_str)

    out = {
        'name': func_name,
        'imports': func_imports,
        'inputs': func_ins,
        'outputs': func_outs,
    }
    return out


def get_snippet_call(package_name, script_name):
    sig = get_snippet_signature(package_name, script_name)
    outs_fmt = ', '.join(sig['outputs'])
    ins_fmt = ', '.join(sig['inputs'])
    ret = f'{sig["name"]}({ins_fmt})'
    if outs_fmt:
        ret = f'{outs_fmt} = {ret}'
    return ret


def get_wrapper_script(package_name, script_name, snippets, outputs):

    ind = '    '
    sigs = [get_snippet_signature(package_name, i['name']) for i in snippets]
    all_ins = [j for i in sigs for j in i['inputs']]
    all_outs = [j for i in sigs for j in i['outputs']]

    for i in outputs:
        if i not in all_outs:
            raise ValueError(f'Cannot output "{i}". No functions return this name.')

    # Required inputs are those that are not output by any snippet
    req_ins = list(set(all_ins) - set(all_outs))
    req_ins_fmt = ', '.join(req_ins)

    main_sig = [f'def main({req_ins_fmt}):']
    main_body = [ind + get_snippet_call(package_name, i['name']) for i in snippets]
    main_outs = ['\n' + ind + f'return {", ".join([i for i in outputs])}']
    main_func = main_sig + main_body + main_outs

    req_imports = [
        'import sys',
        'import hickle',
        'from pathlib import Path',
    ]
    out = req_imports
    out += main_func
    snippet_funcs = '\n'.join([get_snippet(package_name, i['name'], decorator=False)
                               for i in snippets])

    out = '\n'.join(out) + '\n' + snippet_funcs + '\n'
    out += dedent('''\
        if __name__ == '__main__':        
            inputs = hickle.load(sys.argv[1])
            outputs = main(**inputs)
            hickle.dump(outputs, 'outputs.hdf5')

    ''')

    out = autopep8.fix_code(out)
    out = black.format_str(out, mode=black.FileMode())

    return out
