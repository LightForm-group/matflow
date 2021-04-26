"""matflow.models.parameters.py"""

import re
import keyword

import h5py
import hickle

from matflow.utils import zeropad


class Parameters(object):

    """
    Attributes
    ----------
    _element : Element
    _parameters : dict
        Dict mapping the safe names of the parameters to their data indices within the
        HDF5 element_idx group.
    _name_map : dict
        Dict mapping the non-safe names of the parameters to their safe names. A safe name
        refers to a name that can be used as a variable name within Python. For example,
        spaces and dots are removed from non-safe names to become safe names. The reason
        for doing this is to allow the use of dot-notation to access element data/files.

    """

    def __init__(self, element, parameters):

        self._element = element
        self._parameters, self._name_map = self._normalise_params_dict(parameters)

    def __getattr__(self, safe_name):
        if safe_name in self._parameters:
            wkflow = self._element.task.workflow
            names_inv = {safe: non_safe for non_safe, safe in self._name_map.items()}
            name = names_inv[safe_name]
            data_idx = self.get_data_idx(name)
            return wkflow.get_element_data(data_idx)
        else:
            msg = f'{self.__class__.__name__!r} object has no attribute {safe_name!r}.'
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
        out = f'{self.__class__.__name__}({names_fmt})'
        return out

    def _normalise_params_dict(self, parameters):

        normed_data_idx = {}
        name_map = {}
        for name, v in (parameters or {}).items():
            safe_name = self._normalise_param_name(name, normed_data_idx.keys())
            normed_data_idx.update({safe_name: v})
            name_map.update({name: safe_name})

        return normed_data_idx, name_map

    @staticmethod
    def get_element_data_key(element_idx, param_name):
        return f'{zeropad(element_idx, 1000)}_{param_name}'

    @staticmethod
    def _normalise_param_name(param_name, existing_names):
        """Transform a string so that it is a valid Python variable name."""
        param_name_old = param_name
        safe_name = param_name.replace('.', '_dot_').replace(' ', '_').replace('-', '_')
        if (
            re.match(r'\d', safe_name) or
            safe_name in dir(Parameters) or
            keyword.iskeyword(safe_name) or
            safe_name in existing_names
        ):
            safe_name = 'param_' + safe_name

        if re.search(r'[^a-zA-Z0-9_]', safe_name) or not safe_name:
            raise ValueError(f'Invalid parameter name: "{param_name_old}".')

        return safe_name

    def as_dict(self):
        return self.get_parameters(safe_names=False)

    def get_parameters(self, safe_names=True):
        if not safe_names:
            names_inv = {safe: non_safe for non_safe, safe in self._name_map.items()}
            return {names_inv[safe_name]: v for safe_name, v in self._parameters.items()}
        return self._parameters

    def get(self, name, safe_name=False):
        if not safe_name:
            name = self._name_map[name]
        return getattr(self, name)

    def get_all(self, safe_names=False):
        return {
            k: self.get(k, safe_names)
            for k in (self._parameters if safe_names else self._name_map).keys()
        }

    def get_element(self):
        """Not a property to reduce chance of attribute collisions."""
        return self._element

    def get_name_map(self):
        """Not a property to reduce chance of attribute collisions."""
        return self._name_map

    def get_data_idx(self, name, safe_name=False):
        if not safe_name:
            name = self._name_map[name]
        out = self._parameters[name]
        if isinstance(out, list):
            out = tuple(out)
        return out

    def add_parameter(self, name, param_type, value=None, data_idx=None):

        if name in self._name_map:
            raise ValueError(f'Parameter "{name}" already exists.')

        safe_name = self._normalise_param_name(name, self._parameters.keys())
        loaded_path = self._element.task.workflow.loaded_path

        with h5py.File(loaded_path, 'r+') as handle:

            if data_idx is None:
                # Add data to the `element_data` group if required:
                path = '/element_data'
                next_idx = len(handle[path])
                element_data_key = self.get_element_data_key(next_idx, name)
                new_group = handle[path].create_group(element_data_key)
                hickle.dump(value, handle, path=new_group.name)
                data_idx = next_idx

            # Load and save attributes of parameter index dict:
            path = self._element.HDF5_path + f"/'{param_type}_data_idx'"
            attributes = dict(handle[path].attrs)
            param_index = hickle.load(handle, path=path)
            del handle[path]

            # Update and re-dump parameter index dict:
            param_index.update({name: data_idx})
            hickle.dump(param_index, handle, path=path)

            # Update parameter index dict attributes to maintain /workflow_obj loadability
            for k, v in attributes.items():
                handle[path].attrs[k] = v

        self._name_map.update({name: safe_name})
        self._parameters.update({safe_name: data_idx})

        return data_idx


class Files(Parameters):

    def get_lines(self, file_name, lines_slice=(1, 10), safe_name=False):

        if not safe_name:
            file_name = self.get_name_map()[file_name]

        if not isinstance(lines_slice, slice):
            if isinstance(lines_slice, int):
                lines_slice = (lines_slice,)
            lines_slice = slice(*lines_slice)

        return getattr(self, file_name).split('\n')[lines_slice]

    def print_lines(self, file_name, lines_slice=(1, 10), safe_name=False):

        lns = self.get_lines(file_name, lines_slice, safe_name)
        print('\n'.join(lns))
