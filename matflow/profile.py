"""`matflow.profile.py`"""

from pathlib import Path

import yaml

from matflow.errors import ProfileError


def parse_workflow_profile(profile_path):

    with Path(profile_path).open() as handle:
        profile = yaml.safe_load(handle)

    req_keys = ['name', 'tasks']
    task_globals = ['run_options', 'stats']
    good_keys = req_keys + task_globals + ['extend']

    miss_keys = list(set(req_keys) - set(profile.keys()))
    bad_keys = list(set(profile.keys()) - set(good_keys))

    if miss_keys:
        miss_keys_fmt = ', '.join([f'"{i}"' for i in miss_keys])
        raise ProfileError(f'Missing keys in profile: {miss_keys_fmt}.')
    if bad_keys:
        bad_keys_fmt = ', '.join([f'"{i}"' for i in bad_keys])
        raise ProfileError(f'Unknown keys in profile: {bad_keys_fmt}.')

    for i in task_globals:
        if i in profile:
            # Add to each task if it has none:
            for idx, task in enumerate(profile['tasks']):
                if i not in task:
                    profile['tasks'][idx][i] = profile[i]

    workflow_dict = {
        'name': profile['name'],
        'tasks': profile['tasks'],
        'extend': profile.get('extend'),
    }

    return workflow_dict
