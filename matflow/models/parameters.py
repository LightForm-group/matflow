'matflow.models.parameters.py'

import re

import h5py
import hickle


class Parameters(object):

    __PY_RESERVED = [
        'and',
        'as',
        'assert',
        'break',
        'class',
        'continue',
        'def',
        'del',
        'elif',
        'else',
        'except',
        'False',
        'finally',
        'for',
        'from',
        'global',
        'if',
        'import',
        'in',
        'is',
        'lambda',
        'locals',
        'None',
        'nonlocal'
        'not',
        'or',
        'pass',
        'raise',
        'return',
        'True',
        'try',
        'while',
        'with',
        'yield',
    ]

    def __init__(self, element, parameters):

        self._element = element
        self._parameters, self._name_map = self._normalise_params_dict(parameters)

    def __getattr__(self, name):
        if self._name_map[name] in self._parameters:
            wkflow = self._element.task.workflow
            data_idx = self.get_data_idx(name)
            return wkflow.get_element_data(data_idx)
        else:
            msg = f'{self.__class__.__name__!r} object has no attribute {name!r}.'
            raise AttributeError(msg)

    def __setattr__(self, name, value):
        if name in ['_element', '_parameters', '_name_map']:
            super().__setattr__(name, value)
        else:
            raise AttributeError

    def __dir__(self):
        return super().__dir__() + list(self._parameters.keys())

    def __repr__(self):
        names_fmt = ', '.join([f'{i!r}' for i in self._parameters.keys()])
        out = (f'{self.__class__.__name__}({names_fmt})')
        return out

    def _normalise_params_dict(self, parameters):

        normed_data_idx = {}
        name_map = {}
        for name, v in (parameters or {}).items():
            name_normed = self._normalise_param_name(name, normed_data_idx.keys())
            normed_data_idx.update({name_normed: v})
            name_map.update({name: name_normed})

        return normed_data_idx, name_map

    @staticmethod
    def _normalise_param_name(param_name, existing_names):
        'Transform a string so that it is a valid Python variable name.'
        param_name_old = param_name
        param_name = param_name.replace('.', '_dot_').replace(' ', '_space_')
        if (
            re.match(r'\d', param_name) or
            param_name in dir(Parameters) or
            param_name in Parameters.__PY_RESERVED or
            param_name in existing_names
        ):
            param_name = 'param_' + param_name

        if re.search(r'[^a-zA-Z0-9_]', param_name) or not param_name:
            raise ValueError(f'Invalid parameter name: "{param_name_old}".')

        return param_name

    def as_dict(self):
        return self.get_parameters(original_names=True)

    def get_parameters(self, original_names=False):
        if original_names:
            name_inv = {v: k for k, v in self._name_map.items()}
            return {name_inv[k]: v for k, v in self._parameters.items()}
        return self._parameters

    def get(self, name):
        return getattr(self, name)

    def get_all(self):
        return {k: self.get(k) for k in self._name_map.keys()}

    def get_element(self):
        'Not a property to reduce chance of attribute collisions.'
        return self._element

    def get_name_map(self):
        'Not a property to reduce chance of attribute collisions.'
        return self._name_map

    def get_data_idx(self, name):
        'Name is original name'
        out = self._parameters[self._name_map[name]]
        if isinstance(out, list):
            out = tuple(out)
        return out

    def add_parameter(self, name, param_type, value=None, data_idx=None):

        if name in self._name_map:
            raise ValueError(f'Parameter "{name}" already exists.')

        name_normed = self._normalise_param_name(name, self._parameters.keys())
        loaded_path = self._element.task.workflow.loaded_path

        with h5py.File(loaded_path, 'r+') as handle:

            if data_idx is None:
                path = '/element_data/data_0'
                next_idx = len(handle[path])
                new_group = handle[path].create_group(str(next_idx))
                new_group.attrs['type'] = [b'dict_item']
                key_type = [str(type(next_idx)).encode('ascii', 'ignore')]
                new_group.attrs['key_type'] = key_type
                hickle.dump({name: value}, handle, path=new_group.name)
                data_idx = next_idx

            path = self._element.HDF5_path
            key = f'\'{param_type}_data_idx\''
            path += '/' + key

            all_data = hickle.load(handle, path)
            del handle[path]
            all_data.update({name: data_idx})

            hickle.dump(all_data, handle, path=path)
            handle[path].attrs['type'] = [b'dict_item']
            handle[path].attrs['key_type'] = [str(type(key)).encode('ascii', 'ignore')]

        self._name_map.update({name: name_normed})
        self._parameters.update({name_normed: data_idx})

        return data_idx
