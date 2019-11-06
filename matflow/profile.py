"""`matflow.profile.py`"""

from pathlib import Path

import yaml

from matflow import CONFIG


def parse_workflow_profile(profile_path):

    with Path(profile_path).open() as handle:
        profile = yaml.safe_load(handle)

    task_res_names = list(set([i['run_options']['resource'] for i in profile['tasks']]))

    # Add any resource connection that features one of the task resources:
    resource_conns = {}
    for i in task_res_names:
        for j in CONFIG['resource_conns']:
            key = (j['source'], j['destination'])
            for k in key:
                if k not in task_res_names:
                    task_res_names.append(k)
            if key not in resource_conns and i in key:
                resource_conns.update({key: j})

    resources = {}
    task_mach_names = []
    for i in task_res_names:
        found_machine = False
        for j in CONFIG['resources']:
            if i == j['name']:
                resources.update({i: j})
                if j['machine'] not in task_mach_names:
                    task_mach_names.append(j['machine'])
                found_machine = True
                break
        if not found_machine:
            raise ValueError(f'Resource named "{i}" was not found.')

    machines = {}
    for i in task_mach_names:
        for j in CONFIG['machines']:
            if i == j['name']:
                machines.update({i: j})
                # Add any sync client machines:
                for k in j.get('sync_client_paths', []):
                    for m in CONFIG['machines']:
                        if k['machine'] == m['name']:
                            machines.update({k['machine']: m})
                break

    workflow_dict = {
        'human_name': profile.get('name'),
        'extends': profile.get('extends'),
        'machines': machines,
        'resources': resources,
        'resource_conns': resource_conns,
        'tasks': profile['tasks'],
    }

    return workflow_dict
