'matflow.models.element.py'

from matflow.models.parameters import Parameters


class Element(object):

    __slots__ = [
        '_task',
        '_element_idx',
        '_inputs',
        '_outputs',
        '_files',
    ]

    def __init__(self, task, element_idx, inputs_data_idx=None, outputs_data_idx=None,
                 files_data_idx=None):

        self._task = task
        self._element_idx = element_idx

        self._inputs = Parameters(self, inputs_data_idx)
        self._outputs = Parameters(self, outputs_data_idx)
        self._files = Parameters(self, files_data_idx)

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

    def as_dict(self):
        'Return attributes dict with preceding underscores removed.'
        self_dict = {k.lstrip('_'): getattr(self, k) for k in self.__slots__}
        self_dict.pop('task')
        self_dict['inputs_data_idx'] = self_dict.pop('inputs').as_dict()
        self_dict['outputs_data_idx'] = self_dict.pop('outputs').as_dict()
        self_dict['files_data_idx'] = self_dict.pop('files').as_dict()
        return self_dict

    def get_input_data_idx(self, input_name):
        return self.inputs.get_data_idx(input_name)

    def get_output_data_idx(self, output_name):
        return self.outputs.get_data_idx(output_name)

    def get_file_data_idx(self, file_name):
        return self.files.get_data_idx(file_name)

    def get_input(self, input_name):
        return getattr(self.inputs, input_name)

    def get_output(self, output_name):
        return getattr(self.outputs, output_name)

    def get_file(self, file_name):
        return getattr(self.files, file_name)

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
