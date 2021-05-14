"""matflow.models.element.py"""

import copy

import hickle
import h5py

from matflow.models.parameters import Parameters, Files


class Element(object):

    __slots__ = [
        '_task',
        '_element_idx',
        '_inputs',
        '_outputs',
        '_files',
        '_resource_usage',
    ]

    def __init__(self, task, element_idx, inputs_data_idx=None, outputs_data_idx=None,
                 files_data_idx=None, resource_usage=None):

        self._task = task
        self._element_idx = element_idx
        self._resource_usage = resource_usage

        self._inputs = Parameters(self, inputs_data_idx)
        self._outputs = Parameters(self, outputs_data_idx)
        self._files = Files(self, files_data_idx)

    def __repr__(self):
        out = (
            f'{self.__class__.__name__}('
            f'inputs={self.inputs!r}, '
            f'outputs={self.outputs!r}, '
            f'files={self.files!r}'
            f')'
        )
        return out

    @property
    def task(self):
        return self._task

    @property
    def element_idx(self):
        return self._element_idx

    @property
    def resource_usage(self):
        return self._resource_usage

    def as_dict(self):
        """Return attributes dict with preceding underscores removed."""
        self_dict = {k.lstrip('_'): getattr(self, k) for k in self.__slots__}
        self_dict.pop('task')
        self_dict['inputs_data_idx'] = self_dict.pop('inputs').as_dict()
        self_dict['outputs_data_idx'] = self_dict.pop('outputs').as_dict()
        self_dict['files_data_idx'] = self_dict.pop('files').as_dict()
        return self_dict

    def get_parameter_data_idx(self, parameter_name):
        try:
            out = self.outputs.get_data_idx(parameter_name)
        except KeyError:
            out = self.inputs.get_data_idx(parameter_name)

        return out

    def get_input_data_idx(self, input_name, safe_name=False):
        return self.inputs.get_data_idx(input_name, safe_name)

    def get_output_data_idx(self, output_name, safe_name=False):
        return self.outputs.get_data_idx(output_name, safe_name)

    def get_file_data_idx(self, file_name, safe_name=False):
        return self.files.get_data_idx(file_name, safe_name)

    def get_input(self, input_name, safe_name=False):
        if not safe_name:
            input_name = self.inputs.get_name_map()[input_name]
        return getattr(self.inputs, input_name)

    def get_output(self, output_name, safe_name=False):
        if not safe_name:
            output_name = self.outputs.get_name_map()[output_name]
        return getattr(self.outputs, output_name)

    def get_file(self, file_name, safe_name=False):
        if not safe_name:
            file_name = self.files.get_name_map()[file_name]
        return getattr(self.files, file_name)

    def get_file_lines(self, file_name, lines_slice=(10,), safe_name=False):
        return self.files.get_lines(file_name, lines_slice, safe_name)

    def print_file_lines(self, file_name, lines_slice=(10,), safe_name=False):
        self.files.print_lines(file_name, lines_slice, safe_name)

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def files(self):
        return self._files

    @property
    def HDF5_path(self):
        return self.task.HDF5_path + f'/\'elements\'/data/data_{self.element_idx}'

    def add_input(self, input_name, value=None, data_idx=None):
        return self.inputs.add_parameter(input_name, 'inputs', value, data_idx)

    def add_output(self, output_name, value=None, data_idx=None):
        return self.outputs.add_parameter(output_name, 'outputs', value, data_idx)

    def add_file(self, file_name, value=None, data_idx=None):
        return self.files.add_parameter(file_name, 'files', value, data_idx)

    def add_resource_usage(self, resource_usage):

        with h5py.File(self.task.workflow.loaded_path, 'r+') as handle:

            # Load and save attributes of parameter index dict:
            path = self.HDF5_path + "/'resource_usage'"
            attributes = dict(handle[path].attrs)
            del handle[path]

            # Dump resource usage:
            hickle.dump(resource_usage, handle, path=path)

            # Update dict attributes to maintain /workflow_obj loadability
            for k, v in attributes.items():
                handle[path].attrs[k] = v

    def get_element_dependencies(self, recurse=False):
        """Get the task/element indices of elements that a given element depends on.

        Parameters
        ----------
        recurse : bool, optional
            If False, only include task/element indices that are direct dependencies of
            the given element. If True, also include task/element indices that indirect
            dependencies of the given element.

        Returns
        -------
        dict of (int : list)
            Dict whose keys are task indices and whose values are lists of element indices
            for a given task.

        Notes
        -----
        For the inverse, see `get_dependent_elements`.

        """

        task = self.task
        workflow = task.workflow
        elem_deps = {}
        for inp_alias, ins in workflow.elements_idx[task.task_idx]['inputs'].items():
            if ins['task_idx'][self.element_idx] is not None:
                dep_elem_idx = ins['element_idx'][self.element_idx]
                # (maybe not needed)
                if ins['task_idx'][self.element_idx] not in elem_deps:
                    elem_deps.update({ins['task_idx'][self.element_idx]: []})
                elem_deps[ins['task_idx'][self.element_idx]].extend(dep_elem_idx)

        if recurse:
            new_elem_deps = copy.deepcopy(elem_deps)
            for task_idx, element_idx in elem_deps.items():
                for element_idx_i in element_idx:
                    element_i = workflow.tasks[task_idx].elements[element_idx_i]
                    add_elem_deps = element_i.get_element_dependencies(recurse=True)
                    for k, v in add_elem_deps.items():
                        if k not in new_elem_deps:
                            new_elem_deps.update({k: []})
                        new_elem_deps[k].extend(v)

            elem_deps = new_elem_deps

        # Remove repeats:
        for k, v in elem_deps.items():
            elem_deps[k] = list(set(v))

        return elem_deps

    def get_dependent_elements(self, recurse=False):
        """Get the task/element indices of elements that depend on a given element.

        Parameters
        ----------
        recurse : bool, optional
            If False, only include task/element indices that depend directly on the given
            element. If True, also include task/element indices that depend indirectly on
            the given element.

        Returns
        -------
        dict of (int : list)
            Dict whose keys are task indices and whose values are lists of element indices
            for a given task.

        Notes
        -----
        For the inverse, see `get_element_dependencies`.

        """

        task = self.task
        workflow = task.workflow
        dep_elems = {}

        for task_idx, elems_idx in enumerate(workflow.elements_idx):
            for inp_alias, ins in elems_idx['inputs'].items():
                if ins.get('task_idx') == task.task_idx:
                    for element_idx, i in enumerate(ins['element_idx']):
                        if self.element_idx in i:
                            if task_idx not in dep_elems:
                                dep_elems.update({task_idx: []})
                            dep_elems[task_idx].append(element_idx)

        if recurse:
            new_dep_elems = copy.deepcopy(dep_elems)
            for task_idx, element_idx in dep_elems.items():
                for element_idx_i in element_idx:
                    element_i = workflow.tasks[task_idx].elements[element_idx_i]
                    add_elem_deps = element_i.get_dependent_elements(recurse=True)
                    for k, v in add_elem_deps.items():
                        if k not in new_dep_elems:
                            new_dep_elems.update({k: []})
                        new_dep_elems[k].extend(v)

            dep_elems = new_dep_elems

        # Remove repeats:
        for k, v in dep_elems.items():
            dep_elems[k] = list(set(v))

        return dep_elems

    def get_parameter_dependency_value(self, parameter_dependency_name):

        workflow = self.task.workflow

        in_tasks = workflow.get_input_tasks(parameter_dependency_name)
        out_tasks = workflow.get_output_tasks(parameter_dependency_name)
        elem_deps = self.get_element_dependencies(recurse=True)

        if parameter_dependency_name in self.task.schema.input_names:
            param_vals = [self.get_input(parameter_dependency_name)]

        elif out_tasks:
            elems = []
            out_tasks_valid = set(out_tasks) & set(elem_deps)
            if not out_tasks_valid:
                msg = (f'Parameter "{parameter_dependency_name}" is not a dependency of '
                       f'given element of task "{self.task.name}".')
                raise ValueError(msg)
            for task_idx in out_tasks_valid:
                for i in elem_deps[task_idx]:
                    elems.append(workflow.tasks[task_idx].elements[i])
            param_vals = [elem.get_output(parameter_dependency_name) for elem in elems]

        elif in_tasks:
            elems = []
            in_tasks_valid = set(in_tasks) & set(elem_deps)
            if not in_tasks_valid:
                msg = (f'Parameter "{parameter_dependency_name}" is not a dependency of '
                       f'given element of task "{self.task.name}".')
                raise ValueError(msg)
            for task_idx in in_tasks_valid:
                for i in elem_deps[task_idx]:
                    elems.append(workflow.tasks[task_idx].elements[i])
            param_vals = [elem.get_input(parameter_dependency_name) for elem in elems]
        else:
            msg = (f'Parameter "{parameter_dependency_name}" is not an input or output '
                   f'parameter for any workflow task.')
            raise ValueError(msg)

        if len(param_vals) == 1:
            param_vals = param_vals[0]

        return param_vals

    def get_dependent_parameter_value(self, dependent_parameter_name):

        workflow = self.task.workflow

        out_tasks = workflow.get_output_tasks(dependent_parameter_name)
        dep_elems = self.get_dependent_elements(recurse=True)

        if dependent_parameter_name in self.task.schema.outputs:
            param_vals = [self.get_output(dependent_parameter_name)]

        elif out_tasks:
            elems = []
            out_tasks_valid = set(out_tasks) & set(dep_elems)
            if not out_tasks_valid:
                msg = (f'Parameter "{dependent_parameter_name}" does not depend on the '
                       f'given element of task "{self.task.name}".')
                raise ValueError(msg)
            for task_idx in out_tasks_valid:
                for i in dep_elems[task_idx]:
                    elems.append(workflow.tasks[task_idx].elements[i])
            param_vals = [elem.get_output(dependent_parameter_name) for elem in elems]
        else:
            msg = (f'Parameter "{dependent_parameter_name}" is not an output parameter '
                   f'for any workflow task.')
            raise ValueError(msg)

        if len(param_vals) == 1:
            param_vals = param_vals[0]

        return param_vals
