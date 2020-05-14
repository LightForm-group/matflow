"""Module containing unit tests on the `hicklable.to_hicklable` function."""

import unittest
from tempfile import TemporaryFile

import numpy as np
import hickle

from matflow.hicklable import to_hicklable


class ConversionTestCase(unittest.TestCase):
    'Tests on `to_hicklable`.'

    def test_built_ins(self):
        'Test expected output for some built-in types.'

        obj = {
            'a': 1,
            'b': 2.0,
            'c': [3, 4, 5.0],
            'd': (6, 7, 8),
            'e': {9, 10, 11},
            'f': {'f1': 1, 'f2': 2},
            'g': 'hello',
        }
        obj_expected = {
            'a': 1,
            'b': 2.0,
            'c': [3, 4, 5.0],
            'd': (6, 7, 8),
            'e': {9, 10, 11},
            'f': {'f1': 1, 'f2': 2},
            'g': 'hello',
        }
        obj_valid = to_hicklable(obj)
        self.assertTrue(obj_valid == obj_expected)

    def test_arrays(self):
        'Test expected output for some arrays.'

        obj = {
            'int_array': np.array([1, 2, 3]),
            'float_array': np.array([3.3, 2.5, -2.1]),
            'bool_array': np.array([1, 0, 0, 1]).astype(bool),
        }
        obj_valid = to_hicklable(obj)
        self.assertTrue(obj_valid == obj)

    def test_object_dict(self):
        'Test expected output for an object with a __dict__ attribute.'

        class myClassObject(object):
            def __init__(self, a=1): self.a = a

        my_class_obj = myClassObject(a=3.5)

        obj = {'my_class_obj': my_class_obj}
        expected_obj = {'my_class_obj': {'a': 3.5}}
        obj_valid = to_hicklable(obj)
        self.assertTrue(obj_valid == expected_obj)

    def test_object_slots(self):
        'Test expected output for an object with a __slots__ attribute.'

        class myClassObject(object):
            __slots__ = ['a']
            def __init__(self, a=1): self.a = a

        my_class_obj = myClassObject(a=3.5)

        obj = {'my_class_obj': my_class_obj}
        expected_obj = {'my_class_obj': {'a': 3.5}}
        obj_valid = to_hicklable(obj)
        self.assertTrue(obj_valid == expected_obj)

    def test_object_dict_slots(self):
        'Test expected output for an object with __dict__ and __slots__ attributes.'

        class myClassObject(object):
            __slots__ = ['a', '__dict__']
            def __init__(self, a=1): self.a = a

        my_class_obj = myClassObject(a=3.5)
        my_class_obj.b = 2

        obj = {'my_class_obj': my_class_obj}
        expected_obj = {'my_class_obj': {'a': 3.5, 'b': 2}}
        obj_valid = to_hicklable(obj)
        self.assertTrue(obj_valid == expected_obj)

    def test_exclude_dict(self):
        'Test specified keys are excluded from a dict.'
        obj = {'a': 1, 'b': 2.0, 'c': 'hello'}
        expected_obj = {'a': 1, 'c': 'hello'}
        obj_valid = to_hicklable(obj, exclude=['b'])
        self.assertTrue(obj_valid == expected_obj)

    def test_exclude_object(self):
        'Test specified keys are excluded from an object.'

        class myClassObject(object):
            def __init__(self, a=1): self.a = a

        my_class_obj = myClassObject(a=3.5)
        my_class_obj.b = 2
        expected_obj = {'b': 2}
        obj_valid = to_hicklable(my_class_obj, exclude=['a'])
        self.assertTrue(obj_valid == expected_obj)

    def test_replace_dict(self):
        'Test specified substrings in keys are found an replaced from a dict.'
        obj = {'a_hi': 1, 'b': 2.0, 'c_hi': 'hello'}
        expected_obj = {'a_bye': 1, 'b': 2.0, 'c_bye': 'hello'}
        obj_valid = to_hicklable(obj, name_replace={'hi': 'bye'})
        self.assertTrue(obj_valid == expected_obj)

    def test_replace_object(self):
        'Test specified substrings in keys are found an replaced from an object.'

        class myClassObject(object):
            def __init__(self, a=1): self.a = a

        my_class_obj = myClassObject(a=3.5)
        my_class_obj.b_hi = 2
        expected_obj = {'a': 3.5, 'b_bye': 2}
        obj_valid = to_hicklable(my_class_obj, name_replace={'hi': 'bye'})
        self.assertTrue(obj_valid == expected_obj)
