"""Module containing explicit tests on generating `element_idx` from tasks and schemas."""

import unittest

from matflow.models import TaskSchema
from matflow.models.construction import (
    validate_task_dict,
    order_tasks,
    get_element_idx,
)

TEST_DATA = {
    'test_1': {
        'description': 'Simple two-task example with nesting between tasks.',
        'software': [
            {
                'name': 'software_1',
                'version': 1,
                'num_cores': [1, 1, 1],
            }
        ],
        'schemas': [
            {
                'name': 'one',
                'outputs': ['p3'],
                'methods': [
                    {
                        'name': 'method_1',
                        'inputs': ['p1[group=default]', 'p2[group=default]'],
                        'implementations': [{'name': 'software_1'}]
                    }
                ],
            },
            {
                'name': 'two',
                'outputs': ['p5'],
                'methods': [
                    {
                        'name': 'method_1',
                        'inputs': ['p3[group=default]', 'p4[group=default]'],
                        'implementations': [{'name': 'software_1'}]
                    }
                ],
            },
        ],
        'tasks': [
            {
                'name': 'one',
                'sequences': [
                    {'name': 'p1', 'vals': [101, 102], 'nest_idx': 0},
                    {'name': 'p2', 'vals': [201, 202], 'nest_idx': 1}
                ],
                'nest': True,
                'method': 'method_1',
                'software': 'software_1',
            },
            {
                'name': 'two',
                'sequences': [
                    {'name': 'p4', 'vals': [401, 402]},
                ],
                'method': 'method_1',
                'software': 'software_1',
            },
        ],
        'tasks_validated_expected': [
            {
                'name': 'one',
                'context': '',
                'run_options': {'num_cores': 1},
                'stats': True,
                'base': None,
                'sequences': [
                    {'name': 'p1', 'vals': [101, 102], 'nest_idx': 0},
                    {'name': 'p2', 'vals': [201, 202], 'nest_idx': 1}
                ],
                'repeats': 1,
                'merge_priority': None,
                'nest': True,
                'method': 'method_1',
                'software': 'software_1',
                'local_inputs': {
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
                'schema': {
                    'name': 'one',
                    'method': 'method_1',
                    'implementation': 'software_1',
                    'inputs': [
                        {'name': 'p1', 'group': 'default', 'context': None},
                        {'name': 'p2', 'group': 'default', 'context': None},
                    ],
                    'outputs': ['p3'],
                }
            },
            {
                'name': 'two',
                'context': '',
                'run_options': {'num_cores': 1},
                'stats': True,
                'base': None,
                'sequences': [
                    {'name': 'p4', 'vals': [401, 402], 'nest_idx': 0},
                ],
                'repeats': 1,
                'merge_priority': None,
                'nest': True,
                'method': 'method_1',
                'software': 'software_1',
                'local_inputs': {
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
                },
                'schema': {
                    'name': 'two',
                    'method': 'method_1',
                    'implementation': 'software_1',
                    'inputs': [
                        {'name': 'p3', 'group': 'default', 'context': None},
                        {'name': 'p4', 'group': 'default', 'context': None},
                    ],
                    'outputs': ['p5'],
                }
            },
        ],
        'dependency_idx_expected': [[], [0]],
        'task_idx_expected': [0, 1],
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
                        'merge_priority': None,
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
                        'group_by': ['p4', 'repeats'],
                        'nest': True,
                        'group_idx': [0, 1, 2, 3, 4, 5, 6, 7],
                        'group_element_idx': [[0], [1], [2], [3], [4], [5], [6], [7]],
                        'num_groups': 8,
                        'group_size': 1,
                        'merge_priority': None,
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
        'software': [
            {
                'name': 'software_1',
                'version': 1,
                'num_cores': [1, 1, 1],
            }
        ],
        'schemas': [
            {
                'name': 'one',
                'outputs': ['p2', 'p4'],
                'methods': [
                    {
                        'name': 'method_1',
                        'inputs': ['p1[group=default]', 'p3[group=default]'],
                        'implementations': [{'name': 'software_1'}]
                    }
                ],
            },
            {
                'name': 'two',
                'outputs': ['p6'],
                'methods': [
                    {
                        'name': 'method_1',
                        'inputs': ['p5[group=default]'],
                        'implementations': [{'name': 'software_1'}]
                    }
                ],
            },
            {
                'name': 'three',
                'outputs': ['p8'],
                'methods': [
                    {
                        'name': 'method_1',
                        'inputs': [
                            'p4[group=group_A]',
                            'p6[group=default]',
                            'p7[group=default]',
                        ],
                        'implementations': [{'name': 'software_1'}]
                    }
                ],
            },
        ],
        'tasks': [
            {
                'name': 'one',
                'software': 'software_1',
                'method': 'method_1',
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
                'software': 'software_1',
                'method': 'method_1',
                'sequences': [
                    {'name': 'p5', 'vals': [501, 502]},
                ],
                'nest': False,
            },
            {
                'name': 'three',
                'software': 'software_1',
                'method': 'method_1',
                'sequences': [
                    {'name': 'p7', 'vals': [701, 702]},
                ],
                'nest': False,
            },
        ],
        'tasks_validated_expected': [
            {
                'name': 'one',
                'context': '',
                'run_options': {'num_cores': 1},
                'stats': True,
                'base': None,
                'sequences': [
                    {'name': 'p1', 'vals': [101, 102], 'nest_idx': 0},
                    {'name': 'p3', 'vals': [301, 302], 'nest_idx': 1},
                ],
                'repeats': 1,
                'merge_priority': None,
                'nest': False,
                'method': 'method_1',
                'software': 'software_1',
                'local_inputs': {
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
                'schema': {
                    'name': 'one',
                    'method': 'method_1',
                    'implementation': 'software_1',
                    'inputs': [
                        {'name': 'p1', 'group': 'default', 'context': None},
                        {'name': 'p3', 'group': 'default', 'context': None},
                    ],
                    'outputs': ['p2', 'p4'],
                }
            },
            {
                'name': 'two',
                'context': '',
                'run_options': {'num_cores': 1},
                'stats': True,
                'base': None,
                'sequences': [
                    {'name': 'p5', 'vals': [501, 502], 'nest_idx': 0},
                ],
                'repeats': 1,
                'merge_priority': None,
                'nest': False,
                'method': 'method_1',
                'software': 'software_1',
                'local_inputs': {
                    'length': 2,
                    'repeats_idx': [0, 0],
                    'groups': {
                        'default': {
                            'group_by': ['p5', 'repeats'],
                            'nest': False
                        },
                    },
                    'inputs': {
                        'p5': {'vals': [501, 502], 'vals_idx': [0, 1]}
                    },
                },
                'schema': {
                    'name': 'two',
                    'method': 'method_1',
                    'implementation': 'software_1',
                    'inputs': [
                        {'name': 'p5', 'group': 'default', 'context': None},
                    ],
                    'outputs': ['p6'],
                }
            },
            {
                'name': 'three',
                'context': '',
                'run_options': {'num_cores': 1},
                'stats': True,
                'base': None,
                'sequences': [
                    {'name': 'p7', 'vals': [701, 702], 'nest_idx': 0},
                ],
                'repeats': 1,
                'merge_priority': None,
                'nest': False,
                'method': 'method_1',
                'software': 'software_1',
                'local_inputs': {
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
                'schema': {
                    'name': 'three',
                    'method': 'method_1',
                    'implementation': 'software_1',
                    'inputs': [
                        {'name': 'p4', 'group': 'group_A', 'context': None},
                        {'name': 'p6', 'group': 'default', 'context': None},
                        {'name': 'p7', 'group': 'default', 'context': None},
                    ],
                    'outputs': ['p8'],
                }
            },
        ],
        'dependency_idx_expected': [[], [], [0, 1]],
        'task_idx_expected': [0, 1, 2],
        'elements_idx_expected': [
            {
                'num_elements': 4,
                'groups': {
                    'default': {
                        'group_by': ['p1', 'p3', 'repeats'],
                        'group_idx': [0, 1, 2, 3],
                        'group_element_idx': [[0], [1], [2], [3]],
                        'num_groups': 4,
                        'group_size': 1,
                        'nest': False,
                        'merge_priority': None,
                    },
                    'user_group_group_A': {
                        'group_by': ['p1'],
                        'group_idx': [0, 0, 1, 1],
                        'group_element_idx': [[0, 1], [2, 3]],
                        'num_groups': 2,
                        'group_size': 2,
                        'nest': False,
                        'merge_priority': None,
                    },
                },
                'inputs': {'p1': {'input_idx': [0, 1, 2, 3]},
                           'p3': {'input_idx': [0, 1, 2, 3]}}
            },
            {
                'num_elements': 2,
                'groups': {
                    'default': {
                        'group_by': ['p5', 'repeats'],
                        'group_idx': [0, 1],
                        'group_element_idx': [[0], [1]],
                        'num_groups': 2,
                        'group_size': 1,
                        'nest': False,
                        'merge_priority': None,
                    },
                },
                'inputs': {'p5': {'input_idx': [0, 1]}},
            },
            {
                'num_elements': 2,
                'groups': {
                    'default': {
                        'group_by': ['p7', 'repeats'],
                        'group_idx': [0, 1],
                        'group_element_idx': [[0], [1]],
                        'num_groups': 2,
                        'group_size': 1,
                        'nest': False,
                        'merge_priority': None,
                    },
                },
                'inputs': {
                    'p7': {'input_idx': [0, 1]},
                    'p4': {'task_idx': 0,
                           'group': 'group_A',
                           'element_idx': [[0, 1], [2, 3]]},
                    'p6': {'task_idx': 1,
                           'group': 'default',
                           'element_idx': [[0], [1]]},
                },
            },
        ],
    },
    'test_3': {
        'description': ('Demonstrate propagation of a user-defined group through '
                        'dependent tasks.'),
        'software': [
            {
                'name': 'software_1',
                'version': 1,
                'num_cores': [1, 1, 1],
            }
        ],
        'schemas': [
            {
                'name': 'one',
                'outputs': ['p3'],
                'methods': [
                    {
                        'name': 'method_1',
                        'inputs': ['p1[group=default]', 'p2[group=default]'],
                        'implementations': [{'name': 'software_1'}]
                    }
                ],
            },
            {
                'name': 'two',
                'outputs': ['p5'],
                'methods': [
                    {
                        'name': 'method_1',
                        'inputs': ['p3[group=default]', 'p4[group=default]'],
                        'implementations': [{'name': 'software_1'}]
                    }
                ],
            },
            {
                'name': 'three',
                'outputs': ['p7'],
                'methods': [
                    {
                        'name': 'method_1',
                        'inputs': ['p5[group=group_A]', 'p6[group=default]'],
                        'implementations': [{'name': 'software_1'}]
                    }
                ],
            },
        ],
        'tasks': [
            {
                'name': 'one',
                'software': 'software_1',
                'method': 'method_1',
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
                'software': 'software_1',
                'method': 'method_1',
                'base': {'p4': 401},
                'nest': True,
            },
            {
                'name': 'three',
                'software': 'software_1',
                'method': 'method_1',
                'sequences': [
                    {'name': 'p6', 'vals': [601, 602]},
                ]
            }
        ],
        'tasks_validated_expected': [
            {
                'name': 'one',
                'context': '',
                'run_options': {'num_cores': 1},
                'stats': True,
                'base': None,
                'sequences': [
                    {'name': 'p1', 'vals': [101, 102], 'nest_idx': 0},
                    {'name': 'p2', 'vals': [201, 202], 'nest_idx': 1},
                ],
                'repeats': 1,
                'merge_priority': None,
                'nest': True,
                'method': 'method_1',
                'software': 'software_1',
                'local_inputs': {
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
                'schema': {
                    'name': 'one',
                    'method': 'method_1',
                    'implementation': 'software_1',
                    'inputs': [
                        {'name': 'p1', 'group': 'default', 'context': None},
                        {'name': 'p2', 'group': 'default', 'context': None},
                    ],
                    'outputs': ['p3'],
                }
            },
            {
                'name': 'two',
                'context': '',
                'run_options': {'num_cores': 1},
                'stats': True,
                'base': {'p4': 401},
                'sequences': None,
                'repeats': 1,
                'merge_priority': None,
                'nest': True,
                'method': 'method_1',
                'software': 'software_1',
                'local_inputs': {
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
                'schema': {
                    'name': 'two',
                    'method': 'method_1',
                    'implementation': 'software_1',
                    'inputs': [
                        {'name': 'p3', 'group': 'default', 'context': None},
                        {'name': 'p4', 'group': 'default', 'context': None},
                    ],
                    'outputs': ['p5'],
                }
            },
            {
                'name': 'three',
                'context': '',
                'run_options': {'num_cores': 1},
                'stats': True,
                'base': None,
                'sequences': [
                    {'name': 'p6', 'vals': [601, 602], 'nest_idx': 0},
                ],
                'repeats': 1,
                'merge_priority': None,
                'nest': True,  # ------------------- Hmmmm..
                'method': 'method_1',
                'software': 'software_1',
                'local_inputs': {
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
                'schema': {
                    'name': 'three',
                    'method': 'method_1',
                    'implementation': 'software_1',
                    'inputs': [
                        {'name': 'p5', 'group': 'group_A', 'context': None},
                        {'name': 'p6', 'group': 'default', 'context': None},
                    ],
                    'outputs': ['p7'],
                }
            },
        ],
        'dependency_idx_expected': [[], [0], [1]],
        'task_idx_expected': [0, 1, 2],
        'elements_idx_expected': [
            {
                'num_elements': 4,
                'groups': {
                    'default': {
                        'group_by': ['p1', 'p2', 'repeats'],
                        'group_idx': [0, 1, 2, 3],
                        'group_element_idx': [[0], [1], [2], [3]],
                        'num_groups': 4,
                        'group_size': 1,
                        'nest': True,
                        'merge_priority': None,
                    },
                    'user_group_group_A': {
                        'group_by': ['p1'],
                        'group_idx': [0, 0, 1, 1],
                        'group_element_idx': [[0, 1], [2, 3]],
                        'num_groups': 2,
                        'group_size': 2,
                        'nest': False,
                        'merge_priority': None,
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
                        'group_by': ['p4', 'repeats'],
                        'group_idx': [0, 1, 2, 3],
                        'group_element_idx': [[0], [1], [2], [3]],
                        'num_groups': 4,
                        'group_size': 1,
                        'nest': True,
                        'merge_priority': None,
                    },
                    'user_group_group_A': {
                        'group_by': ['p1'],
                        'group_idx': [0, 0, 1, 1],
                        'group_element_idx': [[0, 1], [2, 3]],
                        'num_groups': 2,
                        'group_size': 2,
                        'nest': False,
                        'merge_priority': None,
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
                        'group_by': ['p6', 'repeats'],
                        'group_idx': [0, 1],
                        'group_element_idx': [[0], [1]],
                        'num_groups': 2,
                        'group_size': 1,
                        'nest': True,
                        'merge_priority': None,
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


class InitTasksFullTestCase(unittest.TestCase):
    """Test each step of `init_tasks`."""

    def test_validate_task_dict(self):
        """Test validated task dicts are as expected."""

        for test_data in TEST_DATA.values():

            with self.subTest(test_data=test_data):

                all_software = test_data['software']
                all_schemas = test_data['schemas']
                validated_exp = test_data['tasks_validated_expected']

                for idx, task in enumerate(test_data['tasks']):

                    validated = validate_task_dict(
                        task,
                        is_from_file=False,
                        all_software=all_software,
                        all_task_schemas=all_schemas,
                        check_integrity=False,
                    )
                    validated['schema'] = {
                        'name': validated['schema'].name,
                        'method': validated['schema'].method,
                        'implementation': validated['schema'].implementation,
                        'inputs': validated['schema'].inputs,
                        'outputs': validated['schema'].outputs,
                    }

                    # print('validated:')
                    # pprint(validated)

                    # print('expected:')
                    # pprint(validated_exp[idx])

                    self.assertTrue(validated == validated_exp[idx])

    def test_order_tasks(self):
        """Test expected dependency indices are generated from `order_tasks`"""

        for test_data in TEST_DATA.values():

            with self.subTest(test_data=test_data):

                task_lst = []
                for i in test_data['tasks_validated_expected']:
                    i['schema'] = TaskSchema(**i['schema'])
                    task_lst.append(i)

                tasks_ordered, dep_idx_srt = order_tasks(task_lst)
                dep_idx_exp = test_data['dependency_idx_expected']
                self.assertTrue(dep_idx_srt == dep_idx_exp)

                # Check task_idx correct:
                for idx, i in enumerate(test_data['task_idx_expected']):
                    self.assertTrue(i == tasks_ordered[idx]['task_idx'])

    def test_element_idx(self):
        """Test expected element idx from `get_element_idx`."""

        for test_name, test_data in TEST_DATA.items():

            with self.subTest(test_data=test_data):

                task_lst = []
                for idx, i in enumerate(test_data['tasks_validated_expected']):
                    i['schema'] = TaskSchema(**i['schema'])
                    i['task_idx'] = test_data['task_idx_expected'][idx]
                    task_lst.append(i)

                dep_idx = test_data['dependency_idx_expected']
                elem_idx = get_element_idx(task_lst, dep_idx)

                # print(f'calculated ({test_name}):', flush=True)
                # pprint(elem_idx)

                # print(f'expected ({test_name}):', flush=True)
                # pprint(test_data['elements_idx_expected'])

                self.assertTrue(elem_idx == test_data['elements_idx_expected'])
