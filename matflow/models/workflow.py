import copy
from pathlib import Path
from pprint import pprint
from subprocess import run, PIPE
from warnings import warn

import hickle
import numpy as np
import yaml

from matflow import (CONFIG, CURRENT_MACHINE, SOFTWARE, TASK_SCHEMAS, TASK_INPUT_MAP,
                     TASK_OUTPUT_MAP, TASK_FUNC_MAP, COMMAND_LINE_ARG_MAP)
from matflow.models import Task, Machine, Resource, ResourceConnection
from matflow.models.task import (get_schema_dict, combine_base_sequence, TaskSchema,
                                 get_local_inputs)
from matflow.jsonable import to_jsonable
from matflow.utils import parse_times, zeropad
from matflow.errors import (IncompatibleWorkflow, IncompatibleTaskNesting,
                            MissingMergePriority)


class Workflow(object):

    INIT_STATUS = 'pending'

    def __init__(self, tasks, machines, resources, resource_conns, stage_directory=None,
                 human_id=None, status=None, machine_name=None, human_name=None,
                 extend=None, viewer=False, profile_str=None):

        self.human_name = human_name or ''
        self._extend_paths = [str(Path(i).resolve())
                              for i in extend['paths']] if extend else None
        self.extend_nest_idx = extend['nest_idx'] if extend else None
        self._stage_directory = str(stage_directory)
        self.machines, self.resources, self.resource_conns = self._init_resources(
            machines, resources, resource_conns)

        self.machine_name = machine_name or CURRENT_MACHINE
        self.profile_str = profile_str

        try:
            self.resource_name = self._get_resource_name()
        except ValueError:
            warn(f'Could not find resource name for workflow: {human_name}/{human_id}')

        tasks, elements_idx = self._validate_tasks(tasks)
        self.tasks = tasks
        self._elements_idx = elements_idx

        self.status = status or Workflow.INIT_STATUS  # | 'waiting' | 'complete'
        self.human_id = human_id or self._make_human_id()

        if not self.path.is_dir() and not viewer:
            self._write_directories()
            self._write_hpcflow_workflow()

    def _write_directories(self):
        'Generate task and element directories.'

        self.path.mkdir()

        for elems_idx, task in zip(self.elements_idx, self.tasks):

            # Generate task directory:
            task_path = task.get_task_path(self.path)
            task_path.mkdir()

            num_elems = elems_idx['num_elements']
            # Generate element directories:
            for i in range(num_elems):
                task_elem_path = task_path.joinpath(str(zeropad(i, num_elems - 1)))
                task_elem_path.mkdir()

    def _write_hpcflow_workflow(self):
        'Generate an hpcflow workflow file to execute this workflow.'

        command_groups = []
        variables = {}
        for elems_idx, task in zip(self.elements_idx, self.tasks):

            task_path_rel = str(task.get_task_path(self.path).name)

            # `input_vars` are those inputs that appear in the commands:
            fmt_commands, input_vars = task.schema.command_group.get_formatted_commands(
                task.local_inputs['inputs'].keys())

            cmd_line_inputs = {}
            for local_in_name, local_in in task.local_inputs['inputs'].items():
                if local_in_name in input_vars:

                    # Expand values for intra-task nesting:
                    values = [local_in['vals'][i] for i in local_in['vals_idx']]

                    # Format values:
                    fmt_func_scope = COMMAND_LINE_ARG_MAP.get(
                        (task.schema.name, task.schema.method, task.schema.implementation)
                    )
                    fmt_func = None
                    if fmt_func_scope:
                        fmt_func = fmt_func_scope.get(local_in_name)

                    if not fmt_func:
                        # Apply some default formatting.
                        if isinstance(values[0], list):
                            def fmt_func(x): return ' '.join(['{}'.format(i) for i in x])
                        else:
                            def fmt_func(x): return '{}'.format(x)

                    values_fmt = [fmt_func(i) for i in values]

                    # Expand values for inter-task nesting:
                    values_fmt_all = [
                        values_fmt[i]
                        for i in elems_idx['inputs'][local_in_name]['input_idx']
                    ]
                    cmd_line_inputs.update({local_in_name: values_fmt_all})

            num_elems = elems_idx['num_elements']

            task_path = task.get_task_path(self.path)

            for local_in_name, var_name in input_vars.items():

                var_file_name = '{}.txt'.format(var_name)
                variables.update({
                    var_name: {
                        'file_contents': {
                            'path': var_file_name,
                            'expected_multiplicity': 1,
                        },
                        'value': '{}',
                    }
                })

                # Create text file in each element directory for each in `input_vars`:
                for i in range(num_elems):

                    task_elem_path = task_path.joinpath(str(zeropad(i, num_elems - 1)))
                    in_val = cmd_line_inputs[local_in_name][i]

                    var_file_path = task_elem_path.joinpath(var_file_name)
                    with var_file_path.open('w') as handle:
                        handle.write(in_val + '\n')

            sources = task.software_instance.get('sources', [])
            command_groups.extend([
                {
                    'directory': '.',
                    'nesting': 'hold',
                    'commands': [
                        'matflow prepare-task --task-idx={}'.format(task.task_idx)
                    ]
                },
                {
                    'directory': '<<{}_dirs>>'.format(task_path_rel),
                    'nesting': 'hold',
                    'commands': fmt_commands,
                    'sources': sources,
                },
                {
                    'directory': '.',
                    'nesting': 'hold',
                    'commands': [
                        'matflow process-task --task-idx={}'.format(task.task_idx)
                    ]
                },
            ])

            # Add variable for the task directories:
            variables.update({
                '{}_dirs'.format(task_path_rel): {
                    'file_regex': {
                        'pattern': '({}/[0-9]+$)'.format(task_path_rel),
                        'is_dir': True,
                        'group': 0,
                    },
                    'value': '{}',
                }
            })

        hf_data = {
            'scheduler': 'sge',
            'output_dir': 'output',
            'error_dir': 'output',
            'command_groups': command_groups,
            'variables': variables,
        }

        with self.path.joinpath('1.hf.yml').open('w') as handle:
            yaml.safe_dump(hf_data, handle)

    def _validate_tasks(self, tasks):

        # TODO: validate sequences dicts somewhere.

        task_info_lst = []
        software_instances = []
        for task in tasks:

            software_instance = task.get('software_instance')
            if task.get('software'):
                if not software_instance:
                    software_instance = self._get_software_instance(
                        task['software'],
                        task['run_options']['resource'],
                        task['run_options'].get('num_cores', 1),
                    )
                else:
                    software_instance['num_cores'] = [
                        int(i) for i in software_instance['num_cores']]

            software_instances.append(software_instance)
            schema_dict = get_schema_dict(task['name'], task['method'], software_instance)
            schema = TaskSchema(**schema_dict)

            local_inputs = task.get('local_inputs')
            if local_inputs is None:
                local_inputs = get_local_inputs(
                    base=task.get('base'),
                    num_repeats=task.get('num_repeats'),
                    sequences=task.get('sequences'),
                )

            task_info_lst.append({
                'name': task['name'],
                'inputs': schema.inputs,
                'outputs': schema.outputs,
                'length': local_inputs['length'],
                'nest': task.get('nest'),
                'merge_priority': task.get('merge_priority'),
                'schema': schema,
                'local_inputs': local_inputs,
            })

        task_srt_idx, task_info_lst, elements_idx = check_task_compatibility(
            task_info_lst)

        validated_tasks = []
        for idx, i in enumerate(task_srt_idx):

            # Reorder and instantiate task
            task_i = tasks[i]

            task_i.pop('base', None)
            task_i.pop('sequences', None)
            task_i.pop('software', None)

            task_i['nest'] = task_info_lst[idx]['nest']
            task_i['task_idx'] = task_info_lst[idx]['task_idx']
            task_i['merge_priority'] = task_info_lst[idx]['merge_priority']
            task_i['software_instance'] = software_instances[idx]
            task_i['schema'] = task_info_lst[idx]['schema']
            task_i['local_inputs'] = task_info_lst[idx]['local_inputs']

            task_i_obj = Task(**task_i)
            validated_tasks.append(task_i_obj)

        return validated_tasks, elements_idx

    def get_extended_workflows(self):
        if self.extend_paths:
            return [Workflow.load_state(i.parent) for i in self.extend_paths]
        else:
            return None

    @property
    def elements_idx(self):
        return self._elements_idx

    @property
    def extend_paths(self):
        if self._extend_paths:
            return [Path(i) for i in self._extend_paths]
        else:
            return None

    @property
    def stage_directory(self):
        return Path(self._stage_directory)

    def _get_software_instance(self, software_name, resource_name, num_cores):
        """Find a software instance in the software.yml file that matches the software of
        a given task and a given machine."""

        resource = self.resources[resource_name]

        try:
            machines = [resource.machine] + [
                i['machine'] for i in resource.non_cloud_machines]
        except ValueError:
            machines = [resource.machine]

        machine_names = [i.name for i in machines]

        for soft_inst in SOFTWARE:

            if soft_inst['name'] != software_name:
                continue

            if soft_inst['machine'] in machine_names:
                core_range = soft_inst['num_cores']
                all_num_cores = list(range(*core_range)) + [core_range[1]]
                if num_cores in all_num_cores:
                    return soft_inst

        msg = (f'Could not find a suitable software instance for software '
               f'"{software_name}" on machines {machine_names}, using {num_cores} cores.')
        raise ValueError(msg)

    def _init_resources(self, machines, resources, resource_conns):

        machines_no_sync_client = []
        machines_sync_client = []
        for mach_dict in machines.values():
            if mach_dict.get('sync_client_paths'):
                machines_sync_client.append(mach_dict)
            else:
                machines_no_sync_client.append(mach_dict)

        machines_out = {}
        for mach_dict in machines_no_sync_client + machines_sync_client:
            sync_client_paths = [
                {
                    'machine': machines_out[i['machine']],
                    'sync_path': i['sync_path'],
                }
                for i in mach_dict.get('sync_client_paths', [])
            ]
            machines_out.update({
                mach_dict['name']: Machine(
                    name=mach_dict['name'],
                    os_type=mach_dict['os_type'],
                    is_dropbox=mach_dict.get('is_dropbox', False),
                    sync_client_paths=sync_client_paths,
                )
            })

        resources_out = {}
        for res_name, res_dict in resources.items():
            resources_out.update({
                res_name: Resource(
                    name=res_dict['name'],
                    base_path=res_dict['base_path'],
                    machine=machines_out[res_dict['machine']],
                )
            })

        resource_conns_out = {}
        for res_conn_key, res_conn_dict in resource_conns.items():
            resource_conns_out.update({
                res_conn_key: ResourceConnection(
                    source=resources_out[res_conn_dict['source']],
                    destination=resources_out[res_conn_dict['destination']],
                    hostname=res_conn_dict.get('hostname'),
                )
            })

        return machines_out, resources_out, resource_conns_out

    def _get_resource_name(self):
        'Get name or Workflow stage resource.'

        for resource in self.resources.values():
            try:
                non_cloud_mach = resource.non_cloud_machines
            except ValueError:
                if (resource.machine.name == self.machine_name and
                        resource.base_path == self.stage_directory):
                    return resource.name
                else:
                    continue
            for i in non_cloud_mach:
                res_base_path = Path(str(resource.base_path).lstrip('\\'))
                sync_path = Path(i['sync_path']).joinpath(res_base_path)
                if (i['machine'].name == self.machine_name and
                        sync_path.resolve() == self.stage_directory.resolve()):
                    return resource.name

        raise ValueError('Could not find resource name for the Workflow staging area.')

    def _make_human_id(self):
        hid = parse_times('%Y-%m-%d-%H%M%S')[0]
        if self.human_name:
            hid = self.human_name + '_' + hid
        return hid

    @property
    def stage_machine(self):
        return self.machines[self.machine_name]

    @property
    def stage_resource(self):
        return self.resources[self.resource_name]

    @property
    def path(self):
        return Path(self.stage_directory, self.human_id)

    @property
    def path_str(self):
        return str(self.path)

    @property
    def hdf_path(self):
        return self.path.joinpath('workflow.hdf5')

    def save_state(self, path=None):
        """Save state of workflow to an HDF5 file."""
        path = Path(path or self.hdf_path)
        with path.open('w') as handle:
            hickle.dump(to_jsonable(self), handle)

    @classmethod
    def load_state(cls, path, viewer=False, full_path=False):
        """Load state of workflow from an HDF5 file."""
        path = Path(path)
        if not full_path:
            path = path.joinpath('workflow.hdf5')
        with path.open() as handle:
            obj_json = hickle.load(handle)

        extend = None
        if obj_json['_extend_paths']:
            extend = {
                'paths': obj_json['_extend_paths'],
                'nest_idx': obj_json['extend_nest_idx']
            }

        obj = {
            'tasks': obj_json['tasks'],
            'machines': obj_json['machines'],
            'resources': obj_json['resources'],
            'resource_conns': obj_json['resource_conns'],
            'stage_directory': obj_json['_stage_directory'],
            'extend': extend,
            'human_name': obj_json['human_name'],
            'human_id': obj_json['human_id'],
            'status': obj_json['status'],
            'profile_str': obj_json['profile_str'],
        }
        if viewer:
            obj.update({'viewer': True})

        return cls(**obj)

    def proceed(self):
        'Start or continue the Workflow task execution.'

        print('Workflow.proceed')

        # If task is remote or scheduled, instead of executing the commands, generate
        # an hpcflow project, transfer to remote location and submit.
        # Task status will be 'waiting', the output_map will not yet have been run
        # So once we get to a 'waiting' task, we should try to run the output map
        # if successful, can continue with next task. Otherwise, end.

        allpaths = list(self.path.glob('*'))
        print(f'allpaths: {allpaths}')

        for task_idx, task in enumerate(self.tasks):

            print(f'task_idx: {task_idx}')

            if task.status == 'complete':
                continue

            elif task.status == 'waiting':
                # TODO: Run output map
                # TODO: set task.status to 'complete' and update workflow state
                continue

            elif task.status == 'pending':

                self.prepare_task_inputs(task_idx)
                task.initialise_outputs()
                task_path = task.get_task_path(self.path)
                task_path.mkdir()

                all_task_elem_path = []
                for element_idx in range(len(task)):

                    # print(f'i (element idx): {element_idx}')

                    input_props = task.inputs[element_idx]
                    # print('input_props:')
                    # pprint(input_props)

                    # Make a sub-dir for each element:
                    task_elem_path = task_path.joinpath(
                        str(zeropad(element_idx, len(task) - 1)))
                    task_elem_path.mkdir()
                    all_task_elem_path.append(task_elem_path)

                    # Copy any input files to task element directory:
                    input_file_keys = {}
                    for k, v in input_props.items():
                        if k.startswith('_file:'):
                            src_file_path = self.stage_directory.joinpath(v)
                            dst_file_path = task_elem_path.joinpath(src_file_path.name)
                            dst_file_path.write_bytes(src_file_path.read_bytes())
                            dst_path_rel = dst_file_path.relative_to(task_elem_path)
                            input_file_keys.update({k: str(dst_path_rel)})

                    # Rename input property without "_file:" prefix:
                    for k, new_v in input_file_keys.items():
                        new_k = k.split('_file:')[1]
                        input_props.update({new_k: new_v})
                        del input_props[k]

                    if task.schema.is_func:

                        args = copy.deepcopy(input_props)
                        output = TASK_FUNC_MAP[(task.name, task.method)](**args)

                        print('got output from func map: ')
                        pprint(output)

                        if output is not None:
                            # Assume for now that a function outputs only one
                            # "object/label":
                            task.outputs[element_idx][task.schema.outputs[0]] = output

                            print('updated task output:')
                            pprint(task.outputs)

                    elif task.schema.input_map:
                        # Run input map for each element

                        # For this task, get the input map function lookup:
                        in_map_lookup = TASK_INPUT_MAP[
                            (task.name, task.method, task.software)
                        ]

                        # For each input file to be written, invoke the function:
                        for in_map in task.schema.input_map:

                            func = in_map_lookup[in_map['file']]

                            for k_idx in range(len(in_map['inputs'])):
                                if in_map['inputs'][k_idx] in input_file_keys:
                                    new_k = in_map['inputs'][k_idx].split('_file:')[1]
                                    in_map['inputs'][k_idx] = new_k

                            # Filter only those inputs required for this file:
                            in_map_inputs = {
                                key: val for key, val in input_props.items()
                                if key in in_map['inputs']
                            }
                            file_path = task_elem_path.joinpath(in_map['file'])

                            # print('in_map_inputs:')
                            # pprint(in_map_inputs)

                            func(path=file_path, **in_map_inputs)

                            if task.inputs[element_idx].get('_files') is None:
                                task.inputs[element_idx].update({
                                    '_files': {},
                                })

                            # Add input file path as an input for the task
                            task.inputs[element_idx]['_files'].update({
                                in_map['file']: str(file_path)
                            })

                if not task.schema.is_func:

                    if task.is_scheduled:
                        # Make hpcflow project in task directory:
                        task.schema.command_group.prepare_scheduled_execution()
                        task.status = 'waiting'
                    else:
                        # Write execution "jobscript" in task directory:
                        run_commands = []
                        for element_idx in range(len(task)):
                            run_cmd = task.schema.command_group.prepare_direct_execution(
                                task.inputs[element_idx],
                                task.get_resource(self),
                                all_task_elem_path[element_idx],
                                task.software_instance.get('wsl_wrapper'),
                            )
                            run_commands.append(run_cmd)

                    if task.is_remote(self):
                        # Copy task directory to remote resource
                        # TODO:
                        task.status = 'waiting'
                        if task.pause:
                            print('Pausing task.')
                            break
                        if task.is_scheduled:
                            # Submit hpcflow project remotely over SSH
                            pass
                        else:
                            # Execute "jobscript" over SSH. # change perms to executable.
                            msg = 'Remote non-scheduled not yet supported.'
                            raise NotImplementedError(msg)
                    else:
                        if task.pause:
                            print('Pausing task.')
                            break
                        # Execute "jobscript" locally and wait.
                        for i in run_commands:
                            proc = run(i['run_cmd'], shell=True, stdout=PIPE, stderr=PIPE,
                                       cwd=str(task_path.joinpath(i['run_dir'])))

                            print('stdout', proc.stdout.decode())
                            print('stderr', proc.stderr.decode())

                    if task.status == 'waiting':
                        print('Task is waiting, pausing workflow progression.')
                        break

                    else:

                        for element_idx in range(len(task)):

                            if task.schema.output_map:
                                # For this task, get the output map function lookup:
                                out_map_lookup = TASK_OUTPUT_MAP[
                                    (task.name, task.method, task.software)]

                                # For each output to be parsed, invoke the function:
                                for out_map in task.schema.output_map:

                                    func = out_map_lookup[out_map['output']]

                                    # Add generated file paths to outputs:
                                    if task.outputs[element_idx].get('_files') is None:
                                        task.outputs[element_idx].update({
                                            '_files': {},
                                        })

                                    task_elem_path = all_task_elem_path[element_idx]

                                    # Filter only those file paths required for this output:
                                    file_paths = []
                                    for i in out_map['files']:
                                        out_file_path = task_elem_path.joinpath(i)
                                        file_paths.append(out_file_path)
                                        task.outputs[element_idx]['_files'].update({
                                            i: str(out_file_path),
                                        })

                                    output = func(*file_paths)
                                    # print(f'\nresolved output is: {output}\n')
                                    task.outputs[element_idx][out_map['output']] = output

                            # Add output files:
                            for outp in task.schema.outputs:
                                if outp.startswith('_file:'):
                                    # Add generated file paths to outputs:
                                    if task.outputs[element_idx].get('_files') is None:
                                        task.outputs[element_idx].update({
                                            '_files': {},
                                        })
                                    file_base = outp.split('_file:')[1]
                                    task.outputs[element_idx]['_files'].update({
                                        file_base: str(task_elem_path.joinpath(file_base))})

        # for i in self.tasks:
        #     print('task inputs')
        #     pprint(i.inputs)
        #     print('task outputs')
        #     pprint(i.outputs)

        self.save_state()

    def prepare_task_inputs(self, task_idx):
        """Prepare the inputs of a task that is about to be executed, by searching for
        relevant outputs from all previous tasks.

        Parameters
        ----------
        task_idx : int
            The index of the task that is about to be executed.

        """

        cur_task = self.tasks[task_idx]
        cur_task_ins = cur_task.schema.inputs

        # print(f'\nexpand_task_inputs.')
        # print(f'task_idx: {task_idx} cur_task_ins: {cur_task_ins}')

        if task_idx == 0:
            return

        collated_outputs = []
        for inp_idx, inp in enumerate(cur_task_ins):

            # Also search for outputs in tasks from extended workflow:
            ext_workflow_tasks = []
            if self.extend_paths:
                ext_workflow_tasks = [j for i in self.get_extended_workflows()
                                      for j in i.tasks]

            cur_workflow_tasks = [i for i in self.tasks if i.task_idx < task_idx]
            prev_task_list = ext_workflow_tasks + cur_workflow_tasks

            for prev_task in prev_task_list:

                prev_task_outs = prev_task.schema.outputs

                if inp in prev_task_outs:

                    if inp.startswith('_file:'):
                        file_name = inp.split('_file:')[1]
                        file_base = file_name.split('.')[0]
                        file_func_arg = f'{file_base}_path'
                        inp_name = file_func_arg
                        # print(f'file_name: {file_name}')
                        # print(f'file_base: {file_base}')
                        # print(f'file_func_arg: {file_func_arg}')
                        vals = [{file_func_arg: i['_files'][file_name]}
                                for i in prev_task.outputs]

                    else:
                        vals = [{inp: i[inp]} for i in prev_task.outputs]
                        inp_name = inp

                    # print(f'inp {inp} in prev_tasks_outs!')

                    new_in_nest_idx = prev_task.nest_idx
                    if self.extend_nest_idx:
                        if inp in self.extend_nest_idx:
                            new_in_nest_idx = self.extend_nest_idx[inp]
                    new_inputs = {
                        'name': inp_name,
                        'nest_idx': new_in_nest_idx,
                        'vals': vals,
                    }

                    # print(f'new_inputs: \n{new_inputs}')
                    collated_outputs.append(new_inputs)

        # Coerce original task inputs into form suitable for combining
        # with outputs of previous tasks:
        task_inputs = [{
            'name': '_inputs',
            'nest_idx': cur_task.nest_idx,
            'vals': cur_task.inputs
        }]

        # print(f'task_inputs: \n{task_inputs}')

        # print(f'will now combine base sequence with collated_outputs: '
        #       f'\n{collated_outputs}\n and task_inputs: \n{task_inputs}')

        # Now combine collated outputs with next task's inputs according to
        # respective `nest_idx`s to form expanded inputs:
        expanded_ins = combine_base_sequence(collated_outputs + task_inputs)

        # Reassign task inputs:
        cur_task.inputs = expanded_ins

    def prepare_task(self, task_idx):
        'Prepare inputs and run input maps.'

        task = self.tasks[task_idx]
        elems_idx = self.elements_idx[task_idx]
        num_elems = elems_idx['num_elements']
        inputs = [{} for _ in range(num_elems)]

        for input_name, inputs_idx in elems_idx['inputs'].items():
            task_idx = inputs_idx.get('task_idx')
            if task_idx is not None:
                # Input values should be copied from a previous task's `outputs`
                prev_task = self.tasks[task_idx]
                prev_outs = prev_task.outputs

                if not prev_outs:
                    msg = ('Task "{}" does not have the outputs required to parametrise '
                           'the current task: "{}".')
                    raise ValueError(msg.format(prev_task.name, task.name))

                values_all = [prev_outs[i][input_name] for i in inputs_idx['output_idx']]

            else:
                # Input values should be copied from this task's `local_inputs`

                # Expand values for intra-task nesting:
                local_in = task.local_inputs['inputs'][input_name]
                values = [local_in['vals'][i] for i in local_in['vals_idx']]

                # Expand values for inter-task nesting:
                values_all = [values[i] for i in inputs_idx['input_idx']]

            for element, val in zip(inputs, values_all):
                element.update({input_name: val})

        task.inputs = inputs

        in_map_lookup = TASK_INPUT_MAP.get((task.name, task.method, task.software))
        task_path = task.get_task_path(self.path)
        for elem_idx, elem_inputs in zip(range(num_elems), task.inputs):

            task_elem_path = task_path.joinpath(str(zeropad(elem_idx, num_elems - 1)))

            # For each input file to be written, invoke the function:
            for in_map in task.schema.input_map:

                # Filter only those inputs required for this file:
                in_map_inputs = {
                    key: val for key, val in elem_inputs.items()
                    if key in in_map['inputs']
                }
                file_path = task_elem_path.joinpath(in_map['file'])

                # Run input map to generate required input files:
                func = in_map_lookup[in_map['file']]
                func(path=file_path, **in_map_inputs)

        self.save_state()

    def process_task(self, task_idx):
        'Process outputs from an executed task: run output map and save outputs.'

        task = self.tasks[task_idx]
        elems_idx = self.elements_idx[task_idx]
        num_elems = elems_idx['num_elements']
        outputs = [{} for _ in range(num_elems)]

        # For this task, get the output map function lookup:
        out_map_lookup = TASK_OUTPUT_MAP[(task.name, task.method, task.software)]
        task_path = task.get_task_path(self.path)

        for elem_idx in range(num_elems):

            task_elem_path = task_path.joinpath(str(zeropad(elem_idx, num_elems - 1)))

            # For each output to be parsed, invoke the function:
            for out_map in task.schema.output_map:

                # Filter only those file paths required for this output:
                file_paths = []
                for i in out_map['files']:
                    out_file_path = task_elem_path.joinpath(i)
                    file_paths.append(out_file_path)

                func = out_map_lookup[out_map['output']]
                output = func(*file_paths)
                outputs[elem_idx][out_map['output']] = output

        task.outputs = outputs

        self.save_state()


def check_task_compatibility(task_info_lst):
    'Check workflow has no incompatible tasks.'

    """
    TODO: 
        * enforce restriction: any given output from one task may only be used as
            input in _one_ other task.
        * When considering nesting, specify `nest: True | False` on the outputting
            task (not the inputting task). Must do this since
        * when extending a workflow, need to also specify which outputs to extract and
            whether `nest: True | False`.
        * implement new key: `merge_priority: INT 0 -> len(outputting tasks)-1`:
            * must be specified on all outputting tasks if any of them is `nest: False`
            * if not specified (and all are `nest: True`), default will be set as
                randomly range(len(outputting tasks)-1).

    """

    dependency_idx = get_dependency_idx(task_info_lst)
    check_missing_inputs(task_info_lst, dependency_idx)

    # Find the index at which each task must be positioned to satisfy input
    # dependencies, and reorder tasks (and `dependency_idx`!):
    min_idx = [max(i or [0]) + 1 for i in dependency_idx]
    task_srt_idx = np.argsort(min_idx)
    task_info_lst = [task_info_lst[i] for i in task_srt_idx]
    dependency_idx = [[np.argsort(task_srt_idx)[j] for j in dependency_idx[i]]
                      for i in task_srt_idx]

    # Add sorted task idx:
    for idx, i in enumerate(task_info_lst):
        i['task_idx'] = idx

    # Note: when considering upstream tasks for a given downstream task, need to nest
    # according to the upstream tasks' `num_elements`, not their `length`.
    elements_idx = []
    for idx, downstream_tsk in enumerate(task_info_lst):

        # Do any further downstream tasks depend on this task?
        depended_on = False
        for deps_idx in dependency_idx[(idx + 1):]:
            if idx in deps_idx:
                depended_on = True
                break

        if not depended_on and downstream_tsk.get('nest') is not None:
            msg = '`nest` value is specified but not required for task "{}".'
            warn(msg.format(downstream_tsk['name']))

        # Add `nest: True` by default to this task if nesting is not specified, and if at
        # least one further downstream task depends on this task:
        if downstream_tsk.get('nest', None) is None:
            if depended_on:
                downstream_tsk['nest'] = True

        # Add default `merge_priority` of `None`:
        if 'merge_priority' not in downstream_tsk:
            downstream_tsk['merge_priority'] = None

        upstream_tasks = [task_info_lst[i] for i in dependency_idx[idx]]
        num_elements = get_task_num_elements(downstream_tsk, upstream_tasks)
        downstream_tsk['num_elements'] = num_elements

        task_elems_idx = get_task_elements_idx(downstream_tsk, upstream_tasks)

        params_idx = get_input_elements_idx(task_elems_idx, downstream_tsk, task_info_lst)
        elements_idx.append({
            'num_elements': num_elements,
            'inputs': params_idx,
        })

    return list(task_srt_idx), task_info_lst, elements_idx


def check_missing_inputs(task_info_lst, dependency_list):

    for deps_idx, task_info in zip(dependency_list, task_info_lst):

        defined_inputs = list(task_info['local_inputs']['inputs'].keys())
        task_info['schema'].check_surplus_inputs(defined_inputs)

        if deps_idx:
            for j in deps_idx:
                for output in task_info_lst[j]['outputs']:
                    if output in task_info['inputs']:
                        defined_inputs.append(output)

        task_info['schema'].check_missing_inputs(defined_inputs)


def get_dependency_idx(task_info_lst):

    dependency_idx = []
    all_outputs = []
    for task_info in task_info_lst:
        all_outputs.extend(task_info['outputs'])
        output_idx = []
        for input_j in task_info['inputs']:
            for task_idx_k, task_info_k in enumerate(task_info_lst):
                if input_j in task_info_k['outputs']:
                    output_idx.append(task_idx_k)
        else:
            dependency_idx.append(output_idx)

    if len(all_outputs) != len(set(all_outputs)):
        msg = 'Multiple tasks in the workflow have the same output!'
        raise IncompatibleWorkflow(msg)

    # Check for circular dependencies in task inputs/outputs:
    all_deps = []
    for idx, deps in enumerate(dependency_idx):
        for i in deps:
            all_deps.append(tuple(sorted([idx, i])))

    if len(all_deps) != len(set(all_deps)):
        msg = 'Workflow tasks are circularly dependent!'
        raise IncompatibleWorkflow(msg)

    return dependency_idx


def get_task_num_elements(downstream_task, upstream_tasks):

    num_elements = downstream_task['length']

    if upstream_tasks:

        is_nesting_mixed = len(set([i['nest'] for i in upstream_tasks])) > 1

        for i in upstream_tasks:

            if i['merge_priority'] is None and is_nesting_mixed:
                msg = ('`merge_priority` for task "{}" must be specified, because'
                       ' nesting is mixed.')
                raise MissingMergePriority(msg.format(i['name']))

            elif i['merge_priority'] is not None and not is_nesting_mixed:
                msg = ('`merge_priority` value is specified but not required for '
                       'task "{}", because nesting is not mixed.')
                warn(msg.format(downstream_task['name']))

        all_merge_priority = [i.get('merge_priority') or 0 for i in upstream_tasks]
        merging_order = np.argsort(all_merge_priority)

        for i in merging_order:
            task_to_merge = upstream_tasks[i]
            if task_to_merge['nest']:
                num_elements *= task_to_merge['num_elements']
            else:
                if task_to_merge['num_elements'] != num_elements:
                    msg = ('Cannot merge without nesting task "{}" (with {} elements)'
                           ' with task "{}" (with {} elements [during '
                           'merge]).'.format(
                               task_to_merge['name'],
                               task_to_merge['num_elements'],
                               downstream_task['name'],
                               num_elements,
                           ))
                    raise IncompatibleTaskNesting(msg)

    return num_elements


def get_task_elements_idx(downstream_task, upstream_tasks):
    """
    Get the elements indices of upstream task outputs (and those of downstream task local
    inputs) necessary to prepare element inputs for the downstream task.

    Parameters
    ----------
    downstream_task : dict
    upstream_tasks : list of dict

    Returns
    -------
    task_elements_idx : dict of (int: list)
        Dict whose keys are task indices (`task_idx`) and whose values are 1D lists
        corresponding to the element indices from upstream task outputs (or the 
        downstream task local inputs) for each task.

    """

    task_elements_idx = {}

    # First find elements indices for downstream local inputs:
    inputs_idx = np.arange(downstream_task['length'])
    tile_num = downstream_task['length']
    for j in upstream_tasks:
        rep_num = j['num_elements'] if j.get('nest') else 1
        inputs_idx = np.repeat(inputs_idx, rep_num)

    task_elements_idx.update({downstream_task['task_idx']: list(inputs_idx)})

    # Now find element indices for upstream task outputs:
    for idx, i in enumerate(upstream_tasks):

        tile_num_i = 1
        if i.get('nest'):
            tile_num_i = tile_num
            tile_num *= i['num_elements']

        inputs_idx = np.tile(np.arange(i['num_elements']), tile_num_i)

        for j in upstream_tasks[idx + 1:]:
            rep_num = j['num_elements'] if j.get('nest') else 1
            inputs_idx = np.repeat(inputs_idx, rep_num)

        task_elements_idx.update({i['task_idx']: list(inputs_idx)})

    return task_elements_idx


def get_input_elements_idx(task_elements_idx, downstream_task, task_info_lst):

    params_idx = {}
    for input_name in downstream_task['inputs']:
        # Find the task_idx for which this input is an output:
        input_task_idx = None
        for i in task_info_lst:
            if input_name in i['outputs']:
                input_task_idx = i['task_idx']
                param_task_idx = input_task_idx
                break
        if input_task_idx is None:
            input_task_idx = downstream_task['task_idx']
            params_idx.update({
                input_name: {
                    'input_idx': task_elements_idx[input_task_idx],
                }
            })
        else:
            params_idx.update({
                input_name: {
                    'task_idx': param_task_idx,
                    'output_idx': task_elements_idx[input_task_idx],
                }
            })

    return params_idx
