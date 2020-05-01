"""`matflow.profile.py`"""

from pathlib import Path

import yaml

from matflow import CONFIG
from matflow.errors import ProfileError


def parse_workflow_profile(profile_path):

    with Path(profile_path).open() as handle:
        profile = yaml.safe_load(handle)

    req_keys = ['name', 'tasks']
    good_keys = req_keys

    miss_keys = list(set(req_keys) - set(profile.keys()))
    bad_keys = list(set(profile.keys()) - set(good_keys))

    if miss_keys:
        miss_keys_fmt = ', '.join([f'"{i}"' for i in miss_keys])
        raise ProfileError(f'Missing keys in profile: {miss_keys_fmt}.')
    if bad_keys:
        bad_keys_fmt = ', '.join([f'"{i}"' for i in bad_keys])
        raise ProfileError(f'Unknown keys in profile: {bad_keys_fmt}.')

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
        'name': profile['name'],
        'tasks': profile['tasks'],
        'extend': profile.get('extend'),
        'machines': machines,
        'resources': resources,
        'resource_conns': resource_conns,
    }

    return workflow_dict
