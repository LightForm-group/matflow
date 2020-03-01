import copy
from pathlib import Path
from pprint import pprint
from subprocess import run, PIPE
from warnings import warn

import hickle

from matflow import (CONFIG, CURRENT_MACHINE, SOFTWARE, TASK_SCHEMAS, TASK_INPUT_MAP,
                     TASK_OUTPUT_MAP, TASK_FUNC_MAP)
from matflow.models import Task, Machine, Resource, ResourceConnection
from matflow.jsonable import to_jsonable
from matflow.sequence import combine_base_sequence
from matflow.utils import parse_times, zeropad
from matflow.models.task import check_task_compatibility


class Workflow(object):

    INIT_STATUS = 'pending'

    def __init__(self, tasks, machines, resources, resource_conns, stage_directory=None,
                 human_id=None, status=None, machine_name=None, human_name=None,
                 extend=None):

        self.human_name = human_name or ''
        self._extend_paths = [str(Path(i).resolve())
                              for i in extend['paths']] if extend else None
        self.extend_nest_idx = extend['nest_idx'] if extend else None
        self._stage_directory = str(stage_directory)
        self.machines, self.resources, self.resource_conns = self._init_resources(
            machines, resources, resource_conns)

        self.machine_name = machine_name or CURRENT_MACHINE

        try:
            self.resource_name = self._get_resource_name()
        except ValueError:
            warn(f'Could not find resource name for workflow: {human_name}/{human_id}')

        task_objs = []
        for i_idx, i in enumerate(tasks):

            software_instance = i.get('software_instance')
            if i.get('software'):
                if not software_instance:
                    software_instance = self._get_software_instance(
                        i['software'],
                        i['run_options']['resource'],
                        i['run_options'].get('num_cores', 1),
                    )
                else:
                    software_instance['num_cores'] = [
                        int(i) for i in software_instance['num_cores']]

            new_task = Task(
                name=i['name'],
                method=i['method'],
                software_instance=software_instance,
                task_idx=i_idx,
                nest=i['nest'],
                run_options=i['run_options'],
                base=i.get('base'),
                num_repeats=i.get('num_repeats'),
                sequences=i.get('sequences'),
                inputs=i.get('inputs'),
                outputs=i.get('outputs'),
                schema=i.get('schema'),
                status=i.get('status'),
                pause=i.get('pause', False),
            )

            task_objs.append(new_task)

        self.tasks = self._validate_tasks(task_objs)
        self.status = status or Workflow.INIT_STATUS  # | 'waiting' | 'complete'
        self.human_id = human_id or self._make_human_id()

        if not self.path.is_dir():
            self.path.mkdir()

    def _validate_tasks(self, task_objs):

        tasks_compat_props = []
        for i in task_objs:
            tasks_compat_props.append({
                'inputs': i.schema.inputs,
                'outputs': i.schema.outputs,
                'length': len(i),
                'nest_idx': i.nest_idx,
            })

        task_srt_idx, tasks_compat_props = check_task_compatibility(tasks_compat_props)

        # Reorder tasks:
        task_objs = [task_objs[i] for i in task_srt_idx]

        # Add new compatibility properties (num_elements, dependencies) to task objects:
        pass

        return task_objs

    def get_extended_workflows(self):
        if self.extend_paths:
            return [Workflow.load_state(i.parent) for i in self.extend_paths]
        else:
            return None

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

    def save_state(self):
        """Save state of workflow to an HDF5 file."""
        with self.hdf_path.open('w') as handle:
            hickle.dump(to_jsonable(self), handle)

    @classmethod
    def load_state(cls, path):
        """Load state of workflow from an HDF5 file."""
        with Path(path).joinpath('workflow.hdf5').open() as handle:
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
        }
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
