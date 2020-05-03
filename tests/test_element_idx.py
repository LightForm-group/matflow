"""Module containing explicit tests on generating `element_idx` from tasks and schemas."""

import unittest

from matflow.models import TaskSchema
from matflow.models.construction import get_dependency_idx, get_local_inputs

TEST_DATA = {
    'test_1': {
        'description': 'Simple two-task example with nesting between tasks.',
        'schemas': {
            'one': {
                'inputs': {
                    'p1': {'group': 'default'},
                    'p2': {'group': 'default'},
                },
                'outputs': ['p3'],
            },
            'two': {
                'inputs': {
                    'p3': {'group': 'default'},
                    'p4': {'group': 'default'},
                },
                'outputs': ['p5'],
            },
        },
        'tasks': [
            {
                'name': 'one',
                'sequences': [
                    {'name': 'p1', 'vals': [101, 102], 'nest_idx': 0},
                    {'name': 'p2', 'vals': [201, 202], 'nest_idx': 1}
                ],
                'nest': True,
            },
            {
                'name': 'two',
                'sequences': [
                    {'name': 'p4', 'vals': [401, 402]},
                ],
            },
        ],
        'dependency_idx_expected': [[], [0]],
        'local_inputs_expected': [
            {
                'length': 4,
                'repeats_idx': [0, 0, 0, 0],
                'inputs': {
                    'p1': {'vals': [101, 102], 'vals_idx': [0, 0, 1, 1]},
                    'p2': {'vals': [201, 202], 'vals_idx': [0, 1, 0, 1]},
                },
                'groups': {
                    'default': {
                        'group_by': ['p1', 'p2', 'repeats'],
                        'nest': True,
                    },
                },
            },
            {
                'length': 2,
                'repeats_idx': [0, 0],
                'inputs': {
                    'p4': {'vals': [401, 402], 'vals_idx': [0, 1]},
                },
                'groups': {
                    'default': {
                        'group_by': ['p3', 'p4', 'repeats'],
                        'nest': True,
                    },
                },
            }
        ],
        'elements_idx_expected': [
            {
                'num_elements': 4,
                'groups': {
                    'default': {
                        'group_by': ['p1', 'p2', 'repeats'],
                        'nest': True,
                        'group_idx': [0, 1, 2, 3],
                        'group_element_idx': [[0], [1], [2], [3]],
                        'num_groups': 4,
                        'group_size': 1,
                    },
                },
                'inputs': {
                    'p1': {'input_idx': [0, 1, 2, 3]},
                    'p2': {'input_idx': [0, 1, 2, 3]},
                },
            },
            {
                'num_elements': 8,
                'groups': {
                    'default': {
                        'group_by': ['p3', 'p4', 'repeats'],
                        'nest': True,
                        'group_idx': [0, 1, 2, 3, 4, 5, 6, 7],
                        'group_element_idx': [[0], [1], [2], [3], [4], [5], [6], [7]],
                        'num_groups': 8,
                        'group_size': 1,
                    },
                },
                'inputs': {
                    'p3': {
                        'task_idx': 0,
                        'group': 'default',
                        'element_idx': [[0], [1], [2], [3], [0], [1], [2], [3]],
                    },
                    'p4': {'input_idx': [0, 0, 0, 0, 1, 1, 1, 1]}
                },
            },
        ]
    },
    'test_2': {
        'description': ('Two tasks feed into a third task, with one parameter using a '
                        'user-defined group.'),
        'schemas': {
            'one': {
                'inputs': {
                    'p1': {'group': 'default'},
                    'p3': {'group': 'default'}
                },
                'outputs': ['p2', 'p4'],
            },
            'two': {
                'inputs': {
                    'p5': {'group': 'default'},
                },
                'outputs': ['p6'],
            },
            'three': {
                'inputs': {
                    'p4': {'group': 'group_A'},
                    'p6': {'group': 'default'},
                    'p7': {'group': 'default'},
                },
                'outputs': ['p8'],
            },
        },
        'tasks': [
            {
                'name': 'one',
                'sequences': [
                    {'name': 'p1', 'vals': [101, 102], 'nest_idx': 0},
                    {'name': 'p3', 'vals': [301, 302], 'nest_idx': 1},
                ],
                'nest': False,
                'groups': {
                    'group_A': {'group_by': ['p1'], 'nest': False},
                },
            },
            {
                'name': 'two',
                'sequences': [
                    {'name': 'p5', 'vals': [501, 502]},
                ],
                'nest': False,
            },
            {
                'name': 'three',
                'sequences': [
                    {'name': 'p7', 'vals': [701, 702]},
                ],
                'nest': False,
            },
        ],
        'dependency_idx_expected': [[], [], [0, 1]],
        'local_inputs_expected': [
            {
                'length': 4,
                'repeats_idx': [0, 0, 0, 0],
                'groups': {
                    'default': {
                        'group_by': ['p1', 'p3', 'repeats'],
                        'nest': False,
                    },
                    'user_group_group_A': {
                        'group_by': ['p1'],
                        'nest': False,
                    },
                },
                'inputs': {
                    'p1': {'vals': [101, 102], 'vals_idx': [0, 0, 1, 1]},
                    'p3': {'vals': [301, 302], 'vals_idx': [0, 1, 0, 1]},
                },
            },
            {
                'length': 2,
                'repeats_idx': [0, 0],
                'groups': {
                    'default': {
                        'group_by': ['p5', 'repeats'],
                        'nest': False
                    },
                },
                'inputs': {
                    'p5': {'vals': [501, 502], 'vals_idx': [0, 1], }
                },
            },
            {
                'length': 2,
                'repeats_idx': [0, 0],
                'groups': {
                    'default': {
                        'group_by': ['p4', 'p6', 'p7', 'repeats'],
                        'nest': False,
                    },
                },
                'inputs': {
                    'p7': {'vals': [701, 702], 'vals_idx': [0, 1]},
                },
            },
        ],
        'elements_idx_expected': [
            {
                'num_elements': 4,
                'groups': {
                    'default': {
                        'group_idx': [0, 1, 2, 3],
                        'group_element_idx': [[0], [1], [2], [3]],
                        'num_groups': 4,
                        'group_size': 1,
                        'nest': False, },
                    'user_group_group_A': {
                        'group_idx': [0, 0, 1, 1],
                        'group_element_idx': [[0, 1], [2, 3]],
                        'num_groups': 2,
                        'group_size': 2,
                        'nest': False, },
                },
                'inputs': {'p1': {'input_idx': [0, 1, 2, 3]},
                           'p3': {'input_idx': [0, 1, 2, 3]}}
            },
            {
                'num_elements': 2,
                'groups': {'default': {
                    'group_idx': [0, 1],
                    'group_element_idx': [[0], [1]],
                    'num_groups': 2,
                    'group_size': 1,
                    'nest': False, }, },
                'inputs': {'p5': {'input_idx': [0, 1]}},
            },
            {
                'num_elements': 2,
                'groups': {'default': {
                    'group_idx': [0, 1],
                    'group_element_idx': [[0], [1]],
                    'num_groups': 2,
                    'group_size': 1,
                    'nest': False, }, },
                'inputs': {
                    'p7': {'input_idx': [0, 1]},
                    'p4': {'task_idx': 0,
                           'group': 'group_A',
                           'elements_idx': [[0, 1], [2, 3]]},
                    'p6': {'task_idx': 1,
                           'group': 'default',
                           'elements_idx': [[0], [1]]},
                },
            },
        ],
    },
    'test_3': {
        'description': ('Demonstrate propagation of a user-defined group through '
                        'dependent tasks.'),
        'schemas': {
            'one': {
                'inputs': {
                    'p1': {'group': 'default'},
                    'p2': {'group': 'default'},
                },
                'outputs': ['p3'],
            },
            'two': {
                'inputs': {
                    'p3': {'group': 'default'},
                    'p4': {'group': 'default'},
                },
                'outputs': ['p5'],
            },
            'three': {
                'inputs': {
                    'p5': {'group': 'group_A'},
                    'p6': {'group': 'default'},
                },
                'outputs': ['p7'],
            },
        },
        'tasks': [
            {
                'name': 'one',
                'sequences': [
                    {'name': 'p1', 'vals': [101, 102], 'nest_idx': 0},
                    {'name': 'p2', 'vals': [201, 202], 'nest_idx': 1},
                ],
                'nest': True,
                'groups': {
                    'group_A': {'group_by': ['p1'], 'nest': False},
                },
            },
            {
                'name': 'two',
                'base': {'p4': 401},
                'nest': True,
            },
            {
                'name': 'three',
                'sequences': [
                    {'name': 'p6', 'vals': [601, 602]},
                ]
            }
        ],
        'dependency_idx_expected': [[], [0], [1]],
        'local_inputs_expected': [
            {
                'length': 4,
                'repeats_idx': [0, 0, 0, 0],
                'inputs': {
                    'p1': {'vals': [101, 102], 'vals_idx': [0, 0, 1, 1]},
                    'p2': {'vals': [201, 202], 'vals_idx': [0, 1, 0, 1]},
                },
                'groups': {
                    'default': {
                        'group_by': ['p1', 'p2', 'repeats'],
                        'nest': True,
                    },
                    'user_group_group_A': {
                        'group_by': ['p1'],
                        'nest': False,
                    },
                },
            },
            {
                'length': 1,
                'repeats_idx': [0],
                'inputs': {
                    'p4': {'vals': [401], 'vals_idx': [0]}
                },
                'groups': {
                    'default': {
                        'group_by': ['p3', 'p4', 'repeats'],
                        'nest': True,
                    },
                },
            },
            {
                'length': 2,
                'repeats_idx': [0, 0],
                'inputs': {
                    'p6': {'vals': [601, 602], 'vals_idx': [0, 1]},
                },
                'groups': {
                    'default': {
                        'group_by': ['p5', 'p6', 'repeats'],
                        'nest': True,
                    },
                },
            },
        ],
        'elements_idx_expected': [
            {
                'num_elements': 4,
                'groups': {
                    'default': {
                        'group_idx': [0, 1, 2, 3],
                        'group_element_idx': [[0], [1], [2], [3]],
                        'num_groups': 4,
                        'group_size': 1,
                        'nest': True,
                    },
                    'user_group_group_A': {
                        'group_by': ['p1'],
                        'group_idx': [0, 0, 1, 1],
                        'group_element_idx': [[0, 1], [2, 3]],
                        'nest': False,
                    },
                },
                'inputs': {
                    'p1': {'input_idx': [0, 1, 2, 3]},
                    'p2': {'input_idx': [0, 1, 2, 3]},
                },
            },
            {
                'num_elements': 4,
                'groups': {
                    'default': {
                        'group_idx': [0, 1, 2, 3],
                        'group_element_idx': [[0], [1], [2], [3]],
                        'num_groups': 4,
                        'group_size': 1,
                        'nest': True,
                    },
                    'user_group_group_A': {
                        'group_by': ['p1'],
                        'group_idx': [0, 0, 1, 1],
                        'group_element_idx': [[0, 1], [2, 3]],
                        'nest': False,
                    },
                },
                'inputs': {
                    'p3': {'task_idx': 0, 'group': 'default', 'element_idx': [[0], [1], [2], [3]]},
                    'p4': {'input_idx': [0, 0, 0, 0]},
                },
            },
            {
                'num_elements': 2,
                'groups': {
                    'default': {
                        'group_idx': [0, 1],
                        'group_element_idx': [[0], [1]],
                        'num_groups': 2,
                        'group_size': 2,
                        'nest': True,
                    },
                },
                'inputs': {
                    'p5': {'task_idx': 1, 'group': 'group_A', 'element_idx': [[0, 1], [2, 3]]},
                    'p6': {'input_idx': [0, 1]},
                },
            },
        ],
    },
}


def init_schemas(task_lst):
    'Construct TaskSchema objects for test_dependency_idx test, for each example.'
    for idx, i in enumerate(task_lst):
        task_lst[idx]['schema'] = TaskSchema(**i['schema'])
    return task_lst


class ElementIdxFullTestCase(unittest.TestCase):
    'Check each step in generating elements indices is successful.'

    def test_local_inputs(self):
        'Test expected local inputs are generated.'

        for test_data in TEST_DATA.values():

            tasks = test_data['tasks']
            schemas = test_data['schemas']
            local_ins_exp = test_data['local_inputs_expected']

            for idx, task in enumerate(tasks):
                loc_ins = get_local_inputs(
                    list(schemas[task['name']]['inputs'].keys()),
                    base=task.get('base'),
                    sequences=task.get('sequences'),
                    num_repeats=task.get('repeats', 1),
                    groups=task.get('groups'),
                    nest=task.get('nest', True),
                    merge_priority=task.get('merge_priority'),
                )
                self.assertTrue(loc_ins == local_ins_exp[idx])

    def test_dependency_idx(self):
        'Test expected dependency indices are generated.'

        for test_data in TEST_DATA.values():

            tasks = test_data['tasks']
            schemas = test_data['schemas']

            task_info_lst = []
            for task in tasks:
                schema = TaskSchema(
                    name=task['name'],
                    inputs=[{'name': k, **v}
                            for k, v in schemas[task['name']]['inputs'].items()],
                    outputs=schemas[task['name']]['outputs'],
                )
                task_info = {
                    'context': task.get('context', ''),
                    'schema': schema,
                }
                task_info_lst.append(task_info)

            dep_idx_exp = test_data['dependency_idx_expected']
            dep_idx = get_dependency_idx(task_info_lst)
            self.assertTrue(dep_idx == dep_idx_exp)
