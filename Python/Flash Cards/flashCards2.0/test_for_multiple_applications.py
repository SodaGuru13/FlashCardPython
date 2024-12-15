import asyncio
import copy
import keyword
import textwrap
import time
import math
import os
import re
from wasmtime import Store, Module, Instance, Func, FuncType
import inspect
import struct
import traceback
from asyncio import run, create_task, gather
from typing import Callable
import array
import logging
# from log_exception import LogException
from collections import Counter, OrderedDict
from contextlib import contextmanager
from datetime import date
import numpy as np
from ctypes import BigEndianStructure, c_long
from colorama import Fore, Back, Style, init
init(autoreset=True)


class Data:
    def __int__(self):
        return 13


class MyClass:
    x = 5


class Cis:
    def __init__(self) -> None:
        self._x = None

    def getx(self):
        print('Getting value')
        return self._x

    def setx(self, value):
        print('Setting value to ' + value)
        self._x = value

    def delx(self):
        print('Deleting value')
        del self._x

    x = property(getx, setx, delx, "I'm the 'x' property.")


class Parrot:
    def __init__(self):
        self._voltage = 100000

    @property
    def voltage(self):
        """Get the current voltage."""
        return self._voltage


class CProperty:
    def __init__(self):
        self._x = None

    @property
    def x(self):
        """I'm the 'x' property."""
        print('Getting value')
        return self._x

    @x.setter
    def x(self, value):
        print('Setting value to ' + value)
        self._x = value

    @x.deleter
    def x(self):
        print('Deleting value')
        del self._x


class LoggingDict(dict):
    def __setitem__(self, key, value):
        logging.info('Setting to %r' % (key, value))
        super().__setitem__(key, value)


class LoggingOD(LoggingDict, OrderedDict):
    pass


class Root:
    def draw(self):
        # the delegation chain stops here
        assert not hasattr(super(), 'draw')     


class Shape(Root):
    def __init__(self, shapename, **kwds):
        self.shapename = shapename
        super().__init__(**kwds)

    def draw(self):
        print('Drawing. Setting shape to:', self.shapename)
        super().draw()


class ColoredShape(Shape):
    def __init__(self, color, **kwds):
        self.color = color
        super().__init__(**kwds)

    def draw(self):
        print('Drawing. Setting color to:', self.color)
        super().draw()


class Moveable:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self):
        print('Drawing at position:', self.x, self.y)


class MoveableAdapter(Root):
    def __init__(self, x, y, **kwds):
        self.movable = Moveable(x, y)
        super().__init__(**kwds)

    def draw(self):
        self.movable.draw()
        super().draw()


class MovableColoredShape(ColoredShape, MoveableAdapter):
    pass


class OrderedCounter(Counter, OrderedDict):
    'Counter that remembers the order elements are first seen'
    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


class X:
    a = 1


class Student:
    def __init__(self, name):
        self.student_name = name

    def __ne__(self, x):
        # return true for different types
        # of object
        if type(x) != type(self):
            return True

        # return True for different values
        if self.student_name != x.student_name:
            return True
        else:
            return False


class geeks:
    course = 'DSA'

    def purchase(obj):
        print("Purchase course : ", obj.course)


class Method_Student:
    # create a variable
    name = "Geeksforgeeks"

    # create a function
    def print_name(obj):
        print("The name is : ", obj.name)


class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    # a class method to create a
    # Person object by birth year.
    @classmethod
    def fromBirthYear(cls, name, year):
        return cls(name, date.today().year - year)

    def display(self):
        print("Name : ", self.name, "Age : ", self.age)


class Random_Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    @staticmethod
    def from_FathersAge(name, fatherAge, fatherPersonAgeDiff):
        return Person(name, date.today().year - fatherAge + fatherPersonAgeDiff)

    @classmethod
    def from_BirthYear(cls, name, birthYear):
        return cls(name, date.today().year - birthYear)

    def display(self):
        print(self.name + "'s age is: " + str(self.age))


class Man(Random_Person):
    sex = 'Female'


class Old_Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    @classmethod
    def fromBirthYear(cls, name, year):
        return cls(name, date.today().year - year)

    @staticmethod
    def isAdult(age):
        return age > 18


class Static_Method_Test:
    def __init__(self, value):
        self.value = value

    @staticmethod
    def get_max_value(x, y):
        return max(x, y)


class BEPoint(BigEndianStructure):
    _fields_ = [("x", c_long), ("y", c_long)]


class ComplexNumber:
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag

    def __abs__(self):
        # Custom implementation for absolute value computation
        return (self.real**2 + self.imag**2)**0.5


class Vector:
    def __init__(self, components):
        self.components = components

    def __abs__(self):
        # Custom implementation for absolute value computation
        return sum(component**2 for component in self.components)**0.5


class OddCounter:
    def __init__(self, end_range):
        self.start = -1
        if not end_range:
            raise ValueError("end_range value should be specified")
        self.end = end_range

    def __iter__(self):
        return self

    def __next__(self):
        if self.start < self.end-1:
            self.start += 2
            return self.start
        else:
            raise StopIteration


class AsyncOddCounter:
    def __init__(self, end_range):
        self.end = end_range
        self.start = -1

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.start < self.end-1:
            self.start += 2
            return self.start
        else:
            raise StopAsyncIteration


class KeyTaker:
    def __init__(self, keys):
        self.keys = keys

    def __aiter__(self):
        # create an iterator of the input keys
        self.iter_keys = iter(self.keys)
        return self

    async def __anext__(self):
        try:
            # extract the keys one at a time
            k = next(self.iter_keys)
        except StopIteration:
            # raise stopasynciteration at the end of iterator
            raise StopAsyncIteration
        # return values for a key
        value = await all_keys(k)
        return value


class Bool_My_Class():
    def __len__(self):
        return 0


# Python program to illustrate callable()
class GeekObj:
    def __call__(self):
        print('Hello GeeksforGeeks')


# Python program to illustrate callable()
class GeekObjTwo:
    def testFunc(self):
        print('Hello GeeksforGeeks')


class IterCounter:
    def __init__(self, start, end):
        self.num = start
        self.end = end

    def __iter__(self):
        return self

    def __next__(self):
        if self.num > self.end:
            raise StopIteration
        else:
            self.num += 1
            return self.num - 1


class MyRange:
    def __init__(self, start, end):
        self.current = start
        self.end = end

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.end:
            raise StopIteration
        current = self.current
        self.current += 1
        return current


class Context_Manager_Example:
    def __enter__(self):
        print("enter")

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("exit")


class set_env_var:
    def __init__(self, var_name, new_value):
        self.var_name = var_name
        self.new_value = new_value

    def __enter__(self):
        self.original_value = os.environ.get(self.var_name)
        os.environ[self.var_name] = self.new_value

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_value is None:
            del os.environ[self.var_name]
        else:
            os.environ[self.var_name] = self.original_value


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop = time.perf_counter()
        self.elapsed = self.stop - self.start


class LogException:
    def __init__(self, logger, level=logging.ERROR, suppress=False):
        self.logger, self.level, self.suppress = logger, level, suppress

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            info = (exc_type, exc_val, exc_tb)
            self.logger.log(self.level, "Exception occured", exc_info=info)
            return self.suppress
        return False


class ExperimentCopyright:
    __author__ = 'Clayton Buus'
    __copyright__ = f'Copyright {chr(169)} 2024 Clayton Buus.\nAll Rights Reserved.'
    __credit__ = '\tThanks to GeeksForGeeks, Programiz, and W3Schools for providing the ideas for\n\t' \
                 'most of the code outside of this class.'.expandtabs(4)
    __license__ = 'Public Domain'
    __version__ = "1.0"

    def __init__(self):
        print(f'{ExperimentCopyright.__copyright__}\n\n{ExperimentCopyright.__credit__}')


class InputCounter:
    def __missing__(self, key):
        return 0


# this is a class named car
class Car:
    def parts(self):
        pass


# class for getting the class name
class Test:
    @property
    def cls(self):
        return type(self).__name__


class Bike:
    def __init__(self, name, b):
        self.name = name
        self.car = self.Car(b)

    class Car:
        def __init__(self, car):
            self.car = car


class Empty:
    pass


class MyNewBaseClass:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.name = cls.__name__


class MyNewClass(MyNewBaseClass):
    pass


class Geekfgeeks(object):
    def my_method(self):
        pass


class Descriptor(object):
    def __init__(self, name = ''):
        self.name = name

    def __get__(self, obj, objtype):
        return "{}for{}".format(self.name, self.name)

    def __set__(self, obj, name):
        if isinstance(name, str):
            self.name = name
        else:
            raise TypeError("Name should be string")


class GFG(object):
    __name__ = Descriptor()


# Python program to explain property() function
# Alphabet class
class Alphabet:
    def __init__(self, value):
        self._value = value

    # getting the values
    def getValue(self):
        print('Getting value')
        return self._value

    # setting the values
    def setValue(self, value):
        print('Setting value to ' + value)
        self._value = value

    # deleting the values
    def delValue(self):
        print('Deleting value')
        del self._value

    __name__ = property(getValue, setValue, delValue, )


class AtAlphabet:
    def __init__(self, value):
        self._value = value

    # getting the values
    @property
    def __name__(self):
        print('Getting value')
        return self._value

    # setting the values
    @__name__.setter
    def __name__(self, value):
        print('Setting value to ' + value)
        self._value = value

    # deleting the values
    @__name__.deleter
    def __name__(self):
        print('Deleting Value')
        del self._value


class C:
    def f(self):
        pass
    class D:
        def g(self):
            pass


class ValueDescriptor:
    logging.basicConfig(level=logging.INFO)
        
    def __set_name__(self, owner, name):
        self.__qualname__ = f'{owner.__qualname__}.{name}'
        self.private_name = f'_{owner.__qualname__}.{name}'

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        value = getattr(obj, self.private_name)
        logging.info('Accessing %r giving %r', self.__qualname__, value)
        return value

    def __set__(self, obj, value):
        logging.info('Updating %r to %r', self.__qualname__, value)
        setattr(obj, self.private_name, value)


class MyValueClass:
    x = ValueDescriptor()
    
    def __init__(self, x):
        self.x = x


class PowTwo:
    def __init__(self, max=0):
        self.n = 0
        self.max = max
        self.__qualname__ = type(self).__qualname__

    def __iter__(self):
        return self

    def __next__(self):
        if self.n > self.max:
            raise StopIteration
        result = 2 ** self.n
        self.n += 1
        return result


class Coordinate:
    x = 10
    y = -5
    z = 0


# Creation of a simple class with __dir__
# method to demonstrate it's working
class Supermarket:
    # Function __dir__ which list all
    # the base attributes to be used.
    def __dir__(self):
        return ['customer_name', 'product',
                'quantity', 'price', 'date']


class Yeah(object):
    def __init__(self, name):
        self.name = name

    # Gets called when an attribute is accessed
    def __getattribute__(self, item):
        print('__getattribute__', item)
        # Calling the super class to avoid recursion
        return super(Yeah, self).__getattribute__(item)
    
    def __getattr__(self, item):
        print('__getattr__', item)
        return super(Yeah, self).__setattr__(item, 'orphan')


class flow:

    # (using "value: Any" to allow arbitrary type)
    def __understand__(self, name: str, value: ...) -> None: ...


def three_argument_function(x, y, z):
    print(f'{x}, {y}, {z}')


def check_tuple(test_tuple):
    result = type(test_tuple) is tuple
    print('Is variable tuple?: ' + str(result) + ' ' + str(test_tuple))


def describe_bool():
    print(Fore.YELLOW + 'This is a module from Real Python describing the bool class.\n' + Fore.LIGHTMAGENTA_EX
          + '>>True + True + False + True')
    check = True + True + False + True
    print(check)
    print(Fore.LIGHTMAGENTA_EX + '>>issubclass(bool, int)')
    check = issubclass(bool, int)
    print(check)
    print(Fore.LIGHTMAGENTA_EX + '>>isinstance(True, bool)')
    check = isinstance(True, bool)
    print(check)
    print(Fore.LIGHTMAGENTA_EX + '>>isinstance(False, bool)')
    check = isinstance(False, bool)
    print(check)
    print(Fore.LIGHTMAGENTA_EX + '>>isinstance(1, bool)')
    check = isinstance(1, bool)
    print(check)
    print(Fore.LIGHTMAGENTA_EX + '>>isinstance(0, bool)')
    check = isinstance(0, bool)
    print(check)
    print(Fore.LIGHTMAGENTA_EX + '>>import keyword\nkeyword.kwlist')
    print(keyword.kwlist)
    print(Fore.LIGHTMAGENTA_EX + '>>int(True)')
    check = int(True)
    print(check)
    print(Fore.LIGHTMAGENTA_EX + '>>int(False)')
    check = int(False)
    print(check)
    print(Fore.LIGHTMAGENTA_EX + '>>codes = [264, 118, 543, 705, 1152, 674]\n>>N = 3')
    codes = [264, 118, 543, 705, 1152, 674]
    N = 3
    print(Fore.LIGHTMAGENTA_EX + '>>[c % N == 0 for c in codes]')
    codes_true_false_list = [c % N == 0 for c in codes]
    print(codes_true_false_list)
    print(Fore.LIGHTMAGENTA_EX + '>>sum([c % N == 0 for c in codes])')
    check = sum([c % N == 0 for c in codes])
    print(check)


def describe_bytearray():
    print(Fore.YELLOW + 'Bytearray Class Test\n' + Fore.LIGHTMAGENTA_EX + '>>bytearray()')
    byte_array = bytearray()
    print(byte_array)
    print(Fore.LIGHTMAGENTA_EX + '>>bytearray(10)')
    byte_array = bytearray(10)
    print(byte_array)
    print(Fore.LIGHTMAGENTA_EX + '>>bytearray(range(20))')
    byte_array = bytearray(range(20))
    print(byte_array)
    print(Fore.LIGHTMAGENTA_EX + '>>bytearray(b\'Hi!\')')
    byte_array = bytearray(b'Hi!')
    print(byte_array)
    print(Fore.LIGHTMAGENTA_EX + '>>string = "Python is interesting"\n>># string with encoding \'utf-8\'\n'
          + '>>arr = bytearray(string, \'utf-8\')')
    string = "Python is interesting"
    # string with encoding 'utf-8'
    byte_array = bytearray(string, 'utf-8')
    print(byte_array)
    print(Fore.LIGHTMAGENTA_EX + '# Explaining what a iterable looks like as a bytearray.\n>>nums = [1, 2, 3, 4, 5]')
    nums = [1, 2, 3, 4, 5]
    byte_array = bytearray(nums)
    print(byte_array)


def describe_bytes():
    print(Fore.YELLOW + 'Bytes Class Test\n' +  Fore.LIGHTMAGENTA_EX
          + '>>single_quotes = b\'still allows embedded "double" quotes\'\n'
          + '>>double_quotes = b"still allows embeded \'single\' quotes"\n'
          + '>>triple_single_quoted = b\'\'\'3 single quotes\'\'\'\n' + '>>triple_double_quoted = """3 double quotes"""\n'
          + '>>zero_filled = bytes(10)\n' + '>>iterable_integers = bytes(range(20))\n'
          + '>>object_example = bytes(memoryview(iterable_integers))' + '>>string = \'Papa Mónty\''
          + '>>arr = bytes(string, \'utf-8\', errors = \'replace\')')
    single_quotes = b'still allows embedded "double" quotes'
    double_quotes = b"still allows embeded 'single' quotes"
    triple_single_quoted = b'''3 single quotes'''
    triple_double_quoted = b"""3 double quotes"""
    string = 'Papa Mónty'
    arr = bytes(string, 'ascii', errors = 'replace')
    zero_filled = bytes(10)
    iterable_integers = bytes(range(20))
    object_example = bytes(memoryview(iterable_integers))
    print(single_quotes)
    print(double_quotes)
    print(triple_single_quoted)
    print(triple_double_quoted)
    print(zero_filled)
    print(iterable_integers)
    print(object_example)
    print(arr)


def describe_complex():
    print(Fore.YELLOW + 'Complex Class Test\n' + Fore.LIGHTMAGENTA_EX + '>>comp = complex(5, 9)')
    comp = complex(5, 9)
    print(comp)
    print(Fore.LIGHTMAGENTA_EX + '>>comp = complex(\'4 + 7j\')')
    comp = complex('4+7j')
    print(comp)
    print(Fore.LIGHTMAGENTA_EX + '>>comp = complex(4 + 5j, 5 + 4j)')
    comp = complex(4+5j, 5+4j)
    print(comp)
    print(Fore.LIGHTMAGENTA_EX + '>>comp = complex(8)')
    comp = complex(8)
    print(comp)
    print(Fore.LIGHTMAGENTA_EX + '>>comp = complex()')
    comp = complex()
    print(comp)


def describe_dict():
    print(Fore.YELLOW + 'Dict Class Test\n' + Fore.LIGHTMAGENTA_EX
          + '>>a = dict(one=1, two=2, three=3)\n'
          + '>>b = {\'one\': 1, \'two\': 2, \'three\': 3}\n'
          + '>>c = dict(zip([\'one\', \'two\', \'three\'], [1, 2, 3]))\n'
          + '>>d = dict([(\'two\', 2), (\'one\', 1), (\'three\', 3)])\n'
          + '>>e = dict({\'three\': 3, \'one\': 1, \'two\': 2})\n'
          + '>>f = dict({\'one\': 1, \'three\': 3}, two=2)\n'
          + '>>a == b == c == d == e == f')
    a = dict(one=1, two=2, three=3)
    b = {'one': 1, 'two': 2, 'three': 3}
    c = dict(zip(['one', 'two', 'three'], [1, 2, 3]))
    d = dict([('two', 2), ('one', 1), ('three', 3)])
    e = dict({'three': 3, 'one': 1, 'two': 2})
    f = dict({'one': 1, 'three': 3}, two=2)
    bool_value = a == b == c == d == e == f
    print(bool_value)
    print(Fore.LIGHTMAGENTA_EX + '>>key_value = {\'jack\': 4098, \'sjoerd\': 4127}')
    key_value = {'jack': 4098, 'sjoerd': 4127}
    print('key : value')
    for key, value in key_value.items():
        print(key, ':', value)
    print(Fore.LIGHTMAGENTA_EX + '>>value_key = {4098: \'jack\', 4127: \'sjoerd\'}')
    value_key = {4098: 'jack', 4127: 'sjoerd'}
    print('key : value')
    for key, value in value_key.items():
        print(key, ':', value)
    print(Fore.LIGHTMAGENTA_EX + '>>empty = {}')
    empty = {}
    print(empty)
    print(Fore.LIGHTMAGENTA_EX + '>>one_to_ten = {x: x ** 2 for x in range(10)}')
    one_to_ten = {x: x ** 2 for x in range(10)}
    print('key : value')
    for key, value in one_to_ten.items():
        print(key, ':', value)
    print(Fore.LIGHTMAGENTA_EX + '>>empty_dict = dict()')
    empty_dict = dict()
    print(empty_dict)
    print(Fore.LIGHTMAGENTA_EX + '>>tuple_dict = dict([(\'foo\', 100), (\'bar\', 200)])')
    tuple_dict = dict([('foo', 100), ('bar', 200)])
    print('key : value')
    for key, value in tuple_dict.items():
        print(key, ':', value)
    print(Fore.LIGHTMAGENTA_EX + '>>equals_dict = dict(foo=100, bar=200)')
    equals_dict = dict(foo=100, bar=200)
    for key, value in equals_dict.items():
        print(key, ':', value)


def describe_frozenset():
    print(Fore.YELLOW + 'Frozenset Test\n\tFrom https://www.programiz.com/python-programming/methods/built-in/frozenset\n'
          + Fore.LIGHTMAGENTA_EX + '>>vowels = (\'a\', \'e\', \'i\', \'o\', \'u\')\n'
          + '>>fSet = frozenset(vowels)\n>>print(\'The frozen set is:\', fSet)')
    # tuple of vowels
    vowels = ('a', 'e', 'i', 'o', 'u')
    f_set = frozenset(vowels)
    print('The frozen set is:', f_set)
    print(Fore.LIGHTMAGENTA_EX + '>>print(\'The empty frozen set is:\', frozenset())')
    print('The empty frozen set is:', frozenset())
    # frozensets are immutable
    print(Fore.LIGHTMAGENTA_EX + '>>fSet.add(\'v\')\n' + Fore.WHITE + 'Traceback (most recent call last):\n'
          + '\tFile \"<string>, line 8, in <module>\n' + '\t\tfSet.add(\'v\')\n'
          + 'AttributeError: \'frozenset\' object has no attribute \'add\'\n' + Fore.LIGHTMAGENTA_EX
          + '>>person = {"name": "John", "age": 23, "sex": "male"}\n' + '>>fSet = frozenset(person)\n'
          + '>>print(\'The frozen set is:\', fSet)')
    # random dictionary
    person = {"name": "John", "age" : 23, "sex": "male"}
    f_set = frozenset(person)
    print('The frozen set is:', f_set)
    print(Fore.LIGHTMAGENTA_EX + '>>A = frozenset([1, 2, 3, 4])\n>>B = frozenset([3, 4, 5, 6])\n>>Cis = A.copy()\n'
          + '>>print(Cis)')
    # Frozensets
    # initialize A and B
    A = frozenset([1, 2, 3, 4])
    B = frozenset([3, 4, 5, 6])
    # copying a frozenset
    C = A.copy()    # Output: frozenset({1, 2, 3, 4})
    print(C)
    print(Fore.LIGHTMAGENTA_EX + '>>print(A.union(B))')
    # union
    print(A.union(B))   # Output: frozenset({1, 2, 3, 4, 5, 6})
    print(Fore.LIGHTMAGENTA_EX + '>>print(A.intersection(B))')
    # intersection
    print(A.intersection(B))    # Output: frozenset({3, 4})
    print(Fore.LIGHTMAGENTA_EX + '>>print(A.difference(B)')
    # difference
    print(A.difference(B))  # Output: frozenset({1, 2})
    print(Fore.LIGHTMAGENTA_EX + '>>print(A.symmetric_difference(B))')
    # symmetric_difference
    print(A.symmetric_difference(B))    # Output: frozenset({1, 2, 5, 6})
    print(Fore.LIGHTMAGENTA_EX + '>>A = frozenset([1, 2, 3, 4])\n' + '>>B = frozenset([3, 4, 5, 6])\n'
          + '>>Cis = frozenset([5, 6])\n' + '>>print(A.isdisjoint(Cis))')
    # Frozensets
    # initialize A, B and Cis
    A = frozenset([1, 2, 3, 4])
    B = frozenset([3, 4, 5, 6])
    C = frozenset([5, 6])
    # isdisjoint() method
    print(A.isdisjoint(C))  # Output: True
    print(Fore.LIGHTMAGENTA_EX + '>>print(Cis.issubset(B))')
    # issubset() method
    print(C.issubset(B))    # Output: True
    print(Fore.LIGHTMAGENTA_EX + '>>print(B.issuperset(Cis))')
    # issuperset() method
    print(B.issuperset(C))  # Output: True


def describe_int_func():
    print(Fore.YELLOW + 'Int Test\n' + Fore.LIGHTMAGENTA_EX + '>>print(\'int(678.56) is:\', int(678.56))')
    print('int(678.56) is:', int(678.56))
    print(Fore.LIGHTMAGENTA_EX + '>>print(\'int(\'987\') is: \', int(\'987\'))')
    print('int(\'987\') is:', int('987'))
    print(Fore.LIGHTMAGENTA_EX + '>>print(\'int() is:\', int())')
    print('int() is:', int())
    print(Fore.LIGHTMAGENTA_EX + '>>class Data:\n\tdef __int__():\n\t\treturn 13\n'
          + '>>x = Data()\n>>using_int = int(x)\n>>getting_int = x.__int__()\n'
          + '>>print(f\'int(x) returns x.__int__() is: {using_int} returns {getting_int}\')')
    x = Data()
    using_int = int(x)
    getting_int = x.__int__()
    print(f'int(13) returns 13.__int__() is: {using_int} returns {getting_int}')
    print(Fore.LIGHTMAGENTA_EX + '>>x = b\'0016\'\n>>using_int = int(x)\n>>print(using_int)')
    x = b'0016'
    using_int = int(x)
    print(using_int)
    print(Fore.LIGHTMAGENTA_EX + '>>x = bytearray(b\'10\')\n>>using_int = int(x)\n>>print(using_int)')
    x = bytearray(b'10')
    using_int = int(x)
    print(using_int)
    print(Fore.LIGHTMAGENTA_EX + '>>x = \'+156\'\n>>print(int(x))')
    x = '+156'
    print(int(x))
    print(Fore.LIGHTMAGENTA_EX + '>>x = \'-156\'\n>>print(int(x))')
    x = '-156'
    print(int(x))
    print(Fore.LIGHTMAGENTA_EX + '>>x = \'0000000000156\'\n>>print(int(x))')
    x = '0000000000156'
    print(int(x))
    print(Fore.LIGHTMAGENTA_EX + '>>x = \'    156    \'\n>>print(int(x))')
    x = '    156    '
    print(int(x))
    print(Fore.LIGHTMAGENTA_EX + '>>x = \'1_5_6\'\n>>print(int(x))')
    x = '1_5_6'
    print(int(x))
    print(Fore.LIGHTMAGENTA_EX + '>>x = \'d\'\n>>print(int(x, 36))')
    x = 'd'
    print(int(x, 36))
    print(Fore.LIGHTMAGENTA_EX + '>>x = \'V\'\n>>print(int(x, 36))')
    x = 'V'
    print(int(x, 36))
    print(Fore.LIGHTMAGENTA_EX + '>>x = \'0b011\'\n>>print(int(x, 2))')
    x = '0b011'
    print(int(x, 2))
    print(Fore.LIGHTMAGENTA_EX + '>>x = \'0o011\'\n>>print(int(x, 8))')
    x = '0o011'
    print(int(x, 8))
    print(Fore.LIGHTMAGENTA_EX + '>>x = \'0x011\'\n>>print(int(x, 16))')
    x = '0x011'
    print(int(x, 16))
    print(Fore.LIGHTMAGENTA_EX + '>>print(int(\'010\', 8))')
    print(int('010', 8))


def describe_list():
    print(Fore.YELLOW + 'List Test\n' + Fore.LIGHTMAGENTA_EX + '>>stuff = []\n>>print(stuff)')
    stuff = []
    print(stuff)
    print(Fore.LIGHTMAGENTA_EX + '>>stuff = [\'a\']\n>>print(stuff)')
    stuff = ['a']
    print(stuff)
    print(Fore.LIGHTMAGENTA_EX + '>>stuff = [\'a\', \'b\', \'c\']\n>>print(stuff)')
    stuff = ['a', 'b', 'c']
    print(stuff)
    print(Fore.LIGHTMAGENTA_EX + '>>stuff = [\'{0}:00 am\'.format(i + 1) in i in range(11)]\n>>print(stuff)')
    stuff = ['{0}:00 am'.format(i + 1) for i in range(11)]
    print(stuff)
    print(Fore.LIGHTMAGENTA_EX + '>>stuff = list()\n>>print(stuff)')
    stuff = list()
    print(stuff)
    print(Fore.LIGHTMAGENTA_EX + '>>stuff = list(\'stuff\')\n>>print(stuff)')
    stuff = list('stuff')
    print(stuff)
    print(Fore.LIGHTMAGENTA_EX + '>>stuff = list((1, 2, 3))\n>>print(stuff)')
    stuff = list((1, 2, 3))
    print(stuff)


def describe_memory_view():
    print(Fore.YELLOW + 'Memory View Test\n' + Fore.LIGHTMAGENTA_EX + '>>v = memoryview(b\'abcefg\')\n>>v[1]')
    v = memoryview(b'abcefg')
    print(v[1])
    print(Fore.LIGHTMAGENTA_EX + '>>v[-1]')
    print(v[-1])
    print(Fore.LIGHTMAGENTA_EX + '>>v[1:4]')
    print(v[1:4])
    print(Fore.LIGHTMAGENTA_EX + '>>bytes(v[1:4])')
    print(bytes(v[1:4]))
    print(Fore.LIGHTMAGENTA_EX + '>>import array\n>>a = array.array(\'l\', [-11111111, 22222222, -33333333, 44444444])'
          + '\n>>m = memoryview(a)\n>>m[0]')
    a = array.array('l', [-11111111, 22222222, -33333333, 44444444])
    m = memoryview(a)
    print(m[0])
    print(Fore.LIGHTMAGENTA_EX + '>>m[-1]\n' + Fore.WHITE + '{0}\n'.format(m[-1])
          + Fore.LIGHTMAGENTA_EX + '>>m[::2].tolist()\n' + Fore.WHITE + '{0}\n'.format(m[::2].tolist())
          + Fore.LIGHTMAGENTA_EX + '>>data = bytearray(b\'abcefg\'\n>>v = memoryview(data)\n>>v.readonly')
    data = bytearray(b'abcefg')
    v = memoryview(data)
    print('{0}\n'.format(v.readonly) + Fore.LIGHTMAGENTA_EX + '>>v[0] = ord(b\'z\')\n>>data')
    v[0] = ord(b'z')
    print('{0}\n'.format(data) + Fore.LIGHTMAGENTA_EX + '>>v[1:4] = b\'123\'\n>>data')
    v[1:4] = b'123'
    print('{0}\n'.format(data) + Fore.LIGHTMAGENTA_EX
          + '>>v[2:3] = b\'spam\'\n' + Fore.WHITE + 'Traceback (most recent call last):\n'
          + '\tFile "<stdin>", line 1, in <module>\n'
          + 'ValueError: memoryview assignment: lvalue and rvalue have different structures\n'
          + Fore.LIGHTMAGENTA_EX + '>>v[2:6] = b\'spam\'\n>>data')
    v[2:6] = b'spam'
    print('{0}\n'.format(data) + Fore.LIGHTMAGENTA_EX + '>>v = memoryview(b\'abcefg\')')
    v = memoryview(b'abcefg')
    print(Fore.LIGHTMAGENTA_EX + '>>hash(v) = hash(b\'abcefg\')\n' + Fore.WHITE
          + '{0}\n'.format(hash(v) == hash(b'abcefg')) + Fore.LIGHTMAGENTA_EX + '>>hash(v[2:4]) == hash(b\'ce\')\n'
          + Fore.WHITE + '{0}\n'.format(hash(v[2:4]) == hash(b'ce')) + Fore.LIGHTMAGENTA_EX
          + '>>hash(v[::-2]) == hash(b\'abcefg\'[::-2])\n' + Fore.WHITE
          + '{0}'.format(hash(v[::-2]) == hash(b'abcefg'[::-2])))


def describe_object():
    print(Fore.YELLOW + 'Object Test\n\tFound on https://www.w3schools.com/python/python_classes.asp\n'
          + Fore.LIGHTMAGENTA_EX + '>>class MyClass:\n\tx = 5\n>>p1 = MyClass()\n>>print(p1.x)')
    p1 = MyClass()
    print(p1.x)


def describe_property():
    print(Fore.YELLOW + 'Property Test\n\tCombined definition with an example from Geeks for Geeks.\n'
          + Fore.LIGHTMAGENTA_EX + '>>class Cis:\n\tdef __init__(self) -> None:\n\t\tself._x = None\n\n\tdef getx(self):\n\t\t'
          + 'print(\'Getting value\')\n\t\treturn self._x\n\n\tdef setx(self, value):\n\t\t'
          + 'print(\'Setting value to \' + value)\n\t\tself._x = value\n\n\tdef delx(self):\n\t\t'
          + 'print(\'Deleting value\')\n\t\tdel self._x\n\n\t'
          + 'x = property(gets, setx, delx, "I\'m the \'x\' property.")\n>>stuff = Cis()\n>>print(stuff.x)')
    stuff = Cis()
    print(stuff.x)
    print(Fore.LIGHTMAGENTA_EX + '>>stuff.x = \'Headache Time!\'')
    stuff.x = 'Headache Time!'
    print(Fore.LIGHTMAGENTA_EX + '>>del stuff.x')
    del stuff.x
    print(Fore.LIGHTMAGENTA_EX + '>>help(Cis.x)')
    help(Cis.x)
    print(Fore.LIGHTMAGENTA_EX + '>>class Parrot:\n\tdef __init__(self):\n\t\tself.voltage = 100000\n\n\t@property\n\t'
          + 'def voltage(self):\n\t\t"""Get the current voltage."""\n\t\treturn self._voltage\n'
          + '>>parrotstuff = Parrot()\n>>print(parrot_stuff)')
    parrot_stuff = Parrot()
    print(parrot_stuff.voltage)
    print(Fore.LIGHTMAGENTA_EX + '>>help(Parrot.voltage)')
    help(Parrot.voltage)
    print(Fore.LIGHTMAGENTA_EX + '>>class CProperty:\n\tdef __init__(self):\n\t\tself._x = None\n\n\t@property\n\t'
          + 'def x(self):\n\t\tprint(\'Getting value\')\n\t\t"""I\'m the \'x\' property."""\n\t\tself._x\n\n\t'
          + '@x.setter\n\tdef x(self, value):\n\t\tprint(\'Setting value to \' + value)\n\t\tself._x = value\n\n\t'
          + '@x.deleter\n\tdef x(self):\n\t\tprint(\'Deleting value\')\n\t\tdel self._x\n'
          + '>>testproperty = CProperty()\n>>print(testproperty.x)')
    test_property = CProperty()
    print(test_property.x)
    print(Fore.LIGHTMAGENTA_EX + '>>test_property.x = \'Another Headache?!\'')
    test_property.x = 'Another Headache?!'
    print(Fore.LIGHTMAGENTA_EX + '>>del test_property.x')
    del test_property.x
    print(Fore.LIGHTMAGENTA_EX + '>>help(CProperty.x)')
    help(CProperty.x)


def describe_range():
    print(Fore.YELLOW + 'Range Test\n' + Fore.LIGHTMAGENTA_EX + '>>list(range(10))\n' + Fore.WHITE
          + '{0}\n'.format(list(range(10))) + Fore.LIGHTMAGENTA_EX + '>>list(range(1, 11))\n' + Fore.WHITE
          + '{0}\n'.format(list(range(1, 11))) + Fore.LIGHTMAGENTA_EX + '>>list(range(0, 30, 5))\n' + Fore.WHITE
          + '{0}\n'.format(list(range(0, 30, 5))) + Fore.LIGHTMAGENTA_EX + '>>list(range(0, 10, 3))\n' + Fore.WHITE
          + '{0}\n'.format(list(range(0, 10, 3))) + Fore.LIGHTMAGENTA_EX + '>>list(range(0, -10, -1))\n' + Fore.WHITE
          + '{0}\n'.format(list(range(0, -10, -1))) + Fore.LIGHTMAGENTA_EX + '>>list(0)\n' + Fore.WHITE
          + '{0}\n'.format(list(range(0))) + Fore.LIGHTMAGENTA_EX + '>>list(1, 0)\n' + Fore.WHITE
          + '{0}\n'.format(list(range(1, 0))) + Fore.LIGHTMAGENTA_EX + '>>r = range(0, 20, 2)')
    r = range(0, 20, 2)
    print(Fore.LIGHTMAGENTA_EX + '>>r\n' + Fore.WHITE + '{0}\n'.format(r) + Fore.LIGHTMAGENTA_EX + '>>11 in r\n'
          + Fore.WHITE + '{0}\n'.format(11 in r) + Fore.LIGHTMAGENTA_EX + '>>10 in r\n' + Fore.WHITE
          + '{0}\n'.format(10 in r) + Fore.LIGHTMAGENTA_EX + '>>r.index(10)\n' + Fore.WHITE
          + '{0}\n'.format(r.index(10)) + Fore.LIGHTMAGENTA_EX + '>>r[5]\n' + Fore.WHITE + '{0}\n'.format(r[5])
          + Fore.LIGHTMAGENTA_EX + '>>r[:5]\n' + Fore.WHITE + '{0}\n'.format(r[:5]) + Fore.LIGHTMAGENTA_EX
          + '>>r[-1]\n' + Fore.WHITE + '{0}\n'.format(r[-1]) + Fore.LIGHTMAGENTA_EX + '>>range(0) == range(2, 1, 3)\n'
          + Fore.WHITE + '{0}\n'.format(range(0) == range(2, 1, 3)) + Fore.LIGHTMAGENTA_EX
          + '>>range(0, 3, 2) == range(0, 4, 2)\n' + Fore.WHITE + '{0}'.format(range(0, 3, 2) == range(0, 4, 2)))


def describe_set():
    print(Fore.YELLOW + 'Set Test\n' + Fore.LIGHTMAGENTA_EX
          + '>>print(\'Comma Separated Set: {0}\'.format({\'jack\', \'sjoerd\'})')
    print('Comma Separated Set: {0}'.format({'jack', 'sjoerd'}))
    print(Fore.LIGHTMAGENTA_EX + '>>print(\'Set Comprehension: {0}\'.format({c for c in \'abracadabra\' if c not in \'abc\'}))')
    print('Set Comprehension: {0}'.format({c for c in 'abracadabra' if c not in 'abc'}))
    print(Fore.LIGHTMAGENTA_EX + '>>print(\'Type Constructor Empty Set: {0}\'.format(set()))')
    print('Type Constructor Empty Set: {0}'.format(set()))
    print(Fore.LIGHTMAGENTA_EX + '>>print(\'Type Constructor Single Item Set: {0}\'.format(set(\'foobar\')))')
    print('Type Constructor Single Item Set: {0}'.format(set('foobar')))
    print(Fore.LIGHTMAGENTA_EX + '>>print(\'Type Construcor Bracket Set: {0}\'.format(set([\'a\', \'b\', \'foo\'])))')
    print('Type Construcor Bracket Set: {0}'.format(set(['a', 'b', 'foo'])))
    print(Fore.LIGHTMAGENTA_EX
          + '>>print(\'Set of Set: {0}\'.format({\'a\', frozenset([\'a\', \'b\', \'foo\']), \'b\'})')
    print('Set of Set: {0}'.format({'a', frozenset(['a', 'b', 'foo']), 'b'}))


def describe_slice():
    print(Fore.YELLOW + 'Slice Test\n\tFrom https://www.w3schools.com/python/ref_func_slice.asp\n'
          + Fore.LIGHTMAGENTA_EX + '>>a = ("a", "b", "c", "d", "e", "f", "g", "h")\n>>x = slice(2)\n>>print(a[x])')
    a = ("a", "b", "c", "d", "e", "f", "g", "h")
    x = slice(2)
    print(a[x])
    print(Fore.LIGHTMAGENTA_EX + '>>y = slice(3, 5)\n>>print(a[y])')
    y = slice(3, 5)
    print(a[y])
    print(Fore.LIGHTMAGENTA_EX + '>>z = slice(0, 8, 3)\n>>print(a[z])')
    z = slice(0, 8, 3)
    print(a[z])
    print(Fore.LIGHTMAGENTA_EX + '>>print(a[3:7:2])')
    print(a[3:7:2])
    print(Fore.LIGHTMAGENTA_EX + '>>import numpy as np\n>>fly = np.array([[0, 1, 3], [3, 4, 5]])\n>>print(fly[1:2,1])')
    fly = np.array([[0, 1, 3], [3, 4, 5]])
    print(fly[1:2, 1])


def describe_str():
    print(Fore.YELLOW + 'Str Test\n\tFrom https://www.w3schools.com/python/ref_func_str.asp,\n\t'
          + 'https://www.freecodecamp.org/news/python-bytes-to-string-how-to-convert-a-bytestring\n\t/#:~:text=Using%20'
          + 'the%20str()%20constructor,data%20over%20a%20network%20socket.,\n\t'
          + 'and https://www.toppr.com/guides/python-guide/references/methods-and-functions/methods\n\t/built-in/str'
          + '/python-str/\n' + Fore.LIGHTMAGENTA_EX + '>>print(\'{0}\\n\'.format(str()))' + Fore.WHITE
          + '{0}\n'.format(str()) + Fore.LIGHTMAGENTA_EX + '>>x = str(3.5)\n>>print(x)')
    x = str(3.5)
    print(f'{x}\n' + Fore.LIGHTMAGENTA_EX
          + '>>byte_string = b\'Hello, world!\'\n>>string = str(byte_string, encoding=\'utf-8\')\n>>print(string)')
    byte_string = b"Hello, world!"
    string = str(byte_string, encoding='utf-8')
    print(f'{string}\n' + Fore.LIGHTMAGENTA_EX +
          '>>decoded_string = byte_string.decode(\'utf-8\')\n>>print(decoded_string)')
    decoded_string = byte_string.decode('utf-8')
    print(f'{decoded_string}\n' + Fore.LIGHTMAGENTA_EX + '>>a = bytes(\'Americån\', encoding=\'utf-8\')\n'
          + '>>error_string = str(a, encoding=\'ascii\', errors=\'replace\')\n>>print(error_string)')
    a = bytes('Americån', encoding='utf-8')
    error_string = str(a, encoding='ascii', errors='replace')
    print(f'{error_string}\n' + Fore.LIGHTMAGENTA_EX + '>>informal_string = str(b\'Zoot!\')\n>>print(informal_string)')
    informal_string = str(b'Zoot!')
    print(informal_string)


def describe_super():
    print(Fore.YELLOW + 'Super Test\n\tFrom https://rhettinger.wordpress.com/2011/05/26/super-considered-super/\n'
          + Fore.LIGHTMAGENTA_EX + '>>import collections\n'
          + '>>class LoggingDict(dict):\n\tdef __setitem__(self, key, value):\n\t\t'
          + 'logging.info(\'Setting to %r\' % (key, value))\n\t\tsuper().__setitem__(key, value)\n'
          + '>>class LoggingOD(LoggingDict, collections.OrderedDict):\n\tpass\n>>print(LoggingOD.__mro__)')
    print(LoggingOD.__mro__)
    print(Fore.LIGHTMAGENTA_EX + '>>class Root:\n\tdef draw(self):\n\t\t# the delegation chain stops here\n\t\t'
          + 'assert not hasattr(super(), \'draw\')\n>>class Shape(Root):\n\tdef __init__(self, shapename, **kwds):'
          + '\n\t\tself.shapename = shapename\n\t\tsuper().__init__(**kwds)\n\tdef draw(self):\n\t\t'
          + 'print(\'Drawing. Setting shape to:\', self.shapename)\n\t\tsuper().draw()\n>>class ColoredShape(Shape):'
          + '\n\tdef __init__(self, color, **kwds):\n\t\tself.color = color\n\t\tsuper().__init__(**kwds)\n\t'
          + 'def draw(self):\n\t\tprint(\'Drawing. Setting color to:\', self.color)\n\t\tsuper().draw()\n'
          + '>>cs = ColoredShape(color=\'red\', shapename=\'circle\')\n>>cs.draw()')
    cs = ColoredShape(color='red', shapename='circle')
    cs.draw()
    print(Fore.LIGHTMAGENTA_EX + '>>cs = ColoredShape(color=\'blue\', shapename=\'square\')\n>>cs.draw()')
    cs = ColoredShape(color='blue', shapename='square')
    cs.draw()
    print(Fore.LIGHTMAGENTA_EX + '>>class Movable:\n\tdef __init__(self, x, y):\n\t\tself.x = x\n\t\tself.y = y\n\t'
          + 'def draw(self):\n\t\tprint(\'Drawing at position:\', self.x, self.y)\n>>class MoveableAdapter(Root):\n\t'
          + 'def __init__(self, x, y, **kwds):\n\t\tself.movable = Moveable(x, y)\n\t\tsuper().__init__(**kwds)\n\t'
          + 'def draw(self):\n\t\tself.movable.draw()\n\t\tsuper().draw()\n'
          + '>>class MoveableColoredShape(ColoredShape, MovableAdapter):\n\tpass\n'
          + '>>MoveableColoredShape(color=\'red\', shapename=\'triangle\', x=10, y=20).draw()')
    MovableColoredShape(color='red', shapename='triangle', x=10, y=20).draw()
    print(Fore.LIGHTMAGENTA_EX + '>>from collections import Counter, OrderedDict\n'
          + '>>class OrderedCounter(Counter, OrderedDict):\n\t'
          + '\'Counter that remembers the order elements are first seen\'\n\tdef __repr__(self):\n\t\t'
          + 'return \'%s(%r)\' % (self.__class__.__name__, OrderedDict(self))\n\tdef __reduce__(self):\n\t\t'
          + 'return self.__class__, (OrderedDict(self),)\n>>oc = OrderedCounter(\'abracadabra\')>>print(oc)')
    oc = OrderedCounter('abracadabra')
    print(oc)
    print(Fore.LIGHTMAGENTA_EX + '>>position = LoggingOD.__mro__.index\n'
          + '>>assert position(LoggingDict) < position(OrderedDict)\n>>assert position(OrderedDict) < position(dict)')
    position = LoggingOD.__mro__.index
    assert position(LoggingDict) < position(OrderedDict)
    assert position(OrderedDict) < position(dict)


def describe_tuple():
    print(Fore.YELLOW + 'Tuple test\n\tCombination of examples from https://www.dataquest.io/blog/python-tuples/#:~:'
          + '\n\ttext=A%20tuple%20is%20an%20immutable,tuple%20containing%20three%20numeric%20objects and the original'
          + '\n\tdefinition.' + Fore.LIGHTMAGENTA_EX + '>>empty_tuple = ()>>print(empty_tuple)')
    empty_tuple = ()
    print(empty_tuple)
    print(Fore.LIGHTMAGENTA_EX + '>>singleton_tuple_n_p = 1.0,\n>>print(singleton_tuple_n_p)')
    singleton_tuple_n_p = 1.0,
    print(singleton_tuple_n_p)
    print(Fore.LIGHTMAGENTA_EX + '>>singleton_tuple_w_p = (9.9,)\n>>print(singleton_tuple_w_p)')
    singleton_tuple_w_p = (9.9,)
    print(singleton_tuple_w_p)
    print(Fore.LIGHTMAGENTA_EX + '>>triple_tuple_n_p = \'10\', 101, True\n>>print(triple_tuple_n_p)')
    triple_tuple_n_p = '10', 101, True
    print(triple_tuple_n_p)
    print(Fore.LIGHTMAGENTA_EX + '>>triple_tuple_w_p = (\'Casey\', \'Darin\', \'Bella\')\n>>print(triple_tuple_w_p)')
    triple_tuple_w_p = ('Casey', 'Darin', 'Bella')
    print(triple_tuple_w_p)
    print(Fore.LIGHTMAGENTA_EX + '>>print(tuple())\n' + Fore.WHITE + '{0}\n'.format(tuple()) + Fore.LIGHTMAGENTA_EX
          + '>>print(tuple(\'abc\'))\n' + Fore.WHITE + '{0}\n'.format(tuple('abc')) + Fore.LIGHTMAGENTA_EX
          + '>>print(tuple([1, 2, 3])\n' + Fore.WHITE + '{0}\n'.format(tuple([1, 2, 3])) + Fore.LIGHTMAGENTA_EX
          + '>>def three_argument_function(x, y, z):\n\tprint(f\'{x}, {y}, {z}\')\n>>def check_tuple(test_tuple):\n\t'
          + 'result = type(test_tuple) is tuple\n\tprint(\'Is variable tuple?: \' + str(result) + \' \''
          +' + str(test_tuple))\n>>three_argument_function(10, 11, 24)')
    three_argument_function(10, 11, 24)
    print(Fore.LIGHTMAGENTA_EX + '>>check_tuple((10, 11, 24))')
    check_tuple((10, 11, 24))


def describe_type():
    print(Fore.YELLOW + 'Type Test\n' + Fore.LIGHTMAGENTA_EX + '>>class X:\n\ta = 1\n>>X = type(\'X\', (), dict(a=1))\n'
          + '>>print(X)')
    X = type('X', (), dict(a=1))
    print(type(X))
    print(Fore.LIGHTMAGENTA_EX + '>>print(vars(X))')
    print(vars(X))


def describe_bases():
    print(Fore.YELLOW + 'Bases Test\n\tFrom Super Test\n' + Fore.LIGHTMAGENTA_EX
          + '>>class Root:\n\tdef draw(self):\n\t\t# the delegation chain stops here\n\t\t'
          + 'assert not hasattr(super(), \'draw\')\n>>class Shape(Root):\n\tdef __init__(self, shapename, **kwds):'
          + '\n\t\tself.shapename = shapename\n\t\tsuper().__init__(**kwds)\n\tdef draw(self):\n\t\t'
          + 'print(\'Drawing. Setting shape to:\', self.shapename)\n\t\tsuper().draw()\n>>class ColoredShape(Shape):'
          + '\n\tdef __init__(self, color, **kwds):\n\t\tself.color = color\n\t\tsuper().__init__(**kwds)\n\t'
          + 'def draw(self):\n\t\tprint(\'Drawing. Setting color to:\', self.color)\n'
          + '>>print(ColoredShape.__bases__, Shape.__bases__, Root.__bases__)')
    print(ColoredShape.__bases__, Shape.__bases__, Root.__bases__)


def describe_mro():
    print(Fore.YELLOW + 'Mro Test\n\tFrom Super Test\n' + Fore.LIGHTMAGENTA_EX
          + '>>from collections import Counter, OrderedDict\n'
          + '>>class OrderedCounter(Counter, OrderedDict):\n\t'
          + '\'Counter that remembers the order elements are first seen\'\n\tdef __repr__(self):\n\t\t'
          + 'return \'%s(%r)\' % (self.__class__.__name__, OrderedDict(self))\n\tdef __reduce__(self):\n\t\t'
          + 'return self.__class__, (OrderedDict(self),)\n>>print(OrderedCounter.__mro__)')
    print(OrderedCounter.__mro__)
    print(Fore.LIGHTMAGENTA_EX + '>>print(OrderedCounter.mro())')
    print(OrderedCounter.mro())


def describe_subclasses_class():
    print(Fore.YELLOW + 'Subclasses Class Test\n' + Fore.LIGHTMAGENTA_EX
          + '>>int.__subclasses__()\n' + Fore.WHITE + '{0}'.format(int.__subclasses__()))


def describe_not_equal_opterator():
    print(Fore.YELLOW + 'Not Equal Operator Test\n\tFrom https://www.geeksforgeeks.org/python-not-equal-operator/\n'
          + Fore.LIGHTGREEN_EX + '>>A = 1\n>>B = 2\n>>C = 2\n>>print(A!=B)')
    A = 1
    B = 2
    C = 2
    print(str(A != B) + Fore.LIGHTGREEN_EX + '\n>>print(B!=C)\n' + Fore.WHITE + str(B != C) + Fore.LIGHTGREEN_EX
          + '\n>>A = 1\n>>B = 1.0\n>>C = "1"\n>>print(A!=B)')
    A = 1
    B = 1.0
    C = "1"
    print(str(A != B) + Fore.LIGHTGREEN_EX + '\n>>print(B!=C)\n' + Fore.WHITE + str(B != C) + Fore.LIGHTGREEN_EX
          + '\nprint(A!=C)\n' + Fore.WHITE + str(A != C) + Fore.LIGHTGREEN_EX + '\n>>list1 = [10, 20, 30]\n'
          + '>>list2 = [10, 20, 30]\n>>list3 = ["geeks", "for", "geeks"]\n>>print(list1 != list2)')
    list1 = [10, 20, 30]
    list2 = [10, 20, 30]
    list3 = ["geeks", "for", "geeks"]
    print(str(list1 != list2) + Fore.LIGHTGREEN_EX + '\n>>print(list1 != list3)\n' + Fore.WHITE + str(list1 != list3)
          + Fore.LIGHTGREEN_EX + '\n>>str1 = \'Geeks\'\n>>str2 = \'GeeksforGeeks\'\n>>if str1 != str2:\n\t'
          + 'print("Strings are not Equal")\n  else:\n\tprint("Strings are Equal")')
    str1 = 'Geeks'
    str2 = 'GeeksforGeeks'
    if str1 != str2:
        print("Strings are not Equal")
    else:
        print("Strings are Equal")
    print(Fore.LIGHTGREEN_EX + '>>class Student:\n\tdef __init__(self, name):\n\t\tself.student_name = name\n\n\t'
          + 'def __ne__(self, x):\n\t\t# return true for different types\n\t\t# of object\n\t\t'
          + 'if type(x) != type(self):\n\t\t\treturn True\n\t\t# return True for different values\n\t\t'
          + 'if self.student_name != x.student_name:\n\t\t\treturn True\n\t\telse:\n\t\t\treturn False\n'
          + '>>s1 = Student("Shyam")\n>>s2 = Student("Raju")\n>>s3 = Student("babu rao")\n>>print(s1 != s2)')
    s1 = Student("Shyam")
    s2 = Student("Raju")
    s3 = Student("babu rao")
    print(str(s1 != s2) + Fore.LIGHTGREEN_EX + '\n>>print(s2 != s3)\n' + Fore.WHITE + str(s2 != s3))


def describe_unchanged():
    print(Fore.YELLOW + 'Unchanged Test\n' + Fore.LIGHTGREEN_EX + '>>x = 5\n>>print(+x)')
    x = 5
    print(+x)


def describe_negated():
    print(Fore.YELLOW + 'Negated Test\n' + Fore.LIGHTGREEN_EX + '>>x = -8\n>>print(-x)')
    x = -8
    print(-x)


def describe_less_than():
    print(Fore.YELLOW + 'Less Than Test\n\tFrom https://www.tutorialspoint.com/python/python_comparison_operators.htm'
          + Fore.CYAN + '\nBoth Operands Are Integers\n' + Fore.LIGHTGREEN_EX + '>>a = 5\n>>b = 7\n'
          + '>>print("a =",a, "b =",b, "a < b is:", a < b)')
    a = 5
    b = 7
    print("a =", a, "b =", b, "a < b is:", a < b)
    print(Fore.CYAN + 'Comparison Of Int And Float\n' + Fore.LIGHTGREEN_EX + '>>a = 10\n>>b = 10.0\n'
          + '>>print("a =", a, "b =", b, "a < b is:", a < b)')
    a = 10
    b = 10.0
    print("a =", a, "b =", b, "a < b is:", a < b)
    print(Fore.CYAN + 'Comparison Of Complex Numbers\n' + Fore.LIGHTGREEN_EX + '>>a = 10 + 1j\n>>b = 10. - 1j\n'
          + '>>print("a =", a, "b =", b, "a < b is:", a < b)')
    print("TypeError: '<' not supported between instances of 'complex' and 'complex'")
    print(Fore.CYAN + 'Comparison Of Booleans\n' + Fore.LIGHTGREEN_EX + '>>a = True\n>>b = False\n'
          + '>>print("a =", a, "b =", b, "a < b is:", a < b)')
    a = True
    b = False
    print("a =", a, "b =", b, "a < b is:", a < b)
    print(Fore.CYAN + 'Comparison Of Different Sequence Types\n' + Fore.LIGHTGREEN_EX + '>>a = (1, 2, 3)\n'
          + '>>b = [1, 2, 3]>>print("a =", a, "b =", b, "a < b is:", a < b)')
    print("TypeError: '<' not supported between instances of 'tuple' and 'list'")
    print(Fore.CYAN + 'Comparison Of Strings\n' + Fore.LIGHTGREEN_EX + '>>a = \'BAT\'\n>>b = \'BALL\'\n'
          + '>>print("a =", a, "b =", b, "a < b is:", a < b)')
    a = 'BAT'
    b = 'BALL'
    print("a =", a, "b =", b, "a < b is:", a < b)
    print(Fore.CYAN + 'Comparison Of Tuples\n' + Fore.LIGHTGREEN_EX + '>>a = (1, 2, 4)\n>>b = (1, 2, 3)\n'
          + '>>print("a =", a, "b =", b, "a < b is:", a < b)')
    a = (1, 2, 4)
    b = (1, 2, 3)
    print("a =", a, "b =", b, "a < b is:", a < b)
    print(Fore.CYAN + 'Comparison Of Dictionary Objects\n' + Fore.LIGHTGREEN_EX + '>>a = {1:1, 2:2}\n'
          + '>>b = {2:2, 1:1, 3:3}\n>>print("a =", a, "b =", b, "a < b is:", a < b)')
    print("TypeError: '<' not supported between instances of 'dict' and 'dict'")


def describe_less_or_equal():
    print(Fore.YELLOW + 'Less Than Or Equal Test'
          + '\n\tFrom https://www.tutorialspoint.com/python/python_comparison_operators.htm\n' + Fore.CYAN
          + 'Both Operands Are Integers\n' + Fore.LIGHTGREEN_EX + '>>a = 5\n>>b = 7\n'
          + '>>print("a =",a, "b =",b, "a <= b is:", a <= b)')
    a = 5
    b = 7
    print("a =", a, "b =", b, "a <= b is:", a <= b)
    print(Fore.CYAN + 'Comparison Of Int And Float\n' + Fore.LIGHTGREEN_EX + '>>a = 10\n>>b = 10.0\n'
          + '>>print("a =",a, "b =",b, "a <= b is:", a <= b)')
    a = 10
    b = 10.0
    print("a =", a, "b =", b, "a <= b is:", a <= b)
    print(Fore.CYAN + 'Comparison Of Complex Numbers\n' + Fore.LIGHTGREEN_EX + '>>a = 10 + 1j\n>>b = 10. - 1j\n'
          + '>>print("a =",a, "b =",b, "a <= b is:", a <= b)')
    print("TypeError: '<=' not supported between instances of 'complex' and 'complex'")
    print(Fore.CYAN + 'Comparison of Booleans\n' + Fore.LIGHTGREEN_EX + '>>a = True\n>>b = False\n'
          + '>>print("a =",a, "b =",b, "a <= b is:", a <= b)')
    a = True
    b = False
    print("a =", a, "b =", b, "a <= b is:", a <= b)
    print(Fore.CYAN + 'Comparison Of Different Sequence Types\n' + Fore.LIGHTGREEN_EX + '>>a = (1, 2, 3)\n'
          + '>>b = [1, 2, 3]\n>>print("a =",a, "b =",b, "a <= b is:", a <= b)')
    print("TypeError: '<=' not supported between instances of 'tuple' and 'list'")
    print(Fore.CYAN + 'Comparison Of Strings\n' + Fore.LIGHTGREEN_EX + '>>a = \'BAT\'\n>>b = \'BALL\'\n'
          + '>>print("a =",a, "b =",b, "a <= b is:", a <= b)')
    a = 'BAT'
    b = 'BALL'
    print("a =", a, "b =", b, "a <= b is:", a <= b)
    print(Fore.CYAN + 'Comparison Of Tuples\n' + Fore.LIGHTGREEN_EX + '>>a = (1, 2, 4)\n>>b = (1, 2, 3)\n'
          + '>>print("a =",a, "b =",b, "a <= b is:", a <= b)')
    a = (1, 2, 4)
    b = (1, 2, 3)
    print("a =", a, "b =", b, "a <= b is:", a <= b)
    print(Fore.CYAN + 'Comparison Of Dictionary Objects\n' + Fore.LIGHTGREEN_EX + '>>a = {1:1, 2:2}\n'
          + '>>b = {2:2, 1:1, 3:3}\n>>print("a =",a, "b =",b, "a <= b is:", a <= b)')
    print("TypeError: '<=' not supported between instances of 'dict' and 'dict'")


def describe_equal():
    print(Fore.YELLOW + 'Equal Test'
          + '\n\tFrom https://www.tutorialspoint.com/python/python_comparison_operators.htm\n' + Fore.CYAN
          + 'Both Operands Are Integers\n' + Fore.LIGHTGREEN_EX + '>>a = 5\n>>b = 7\n'
          + '>>print("a =",a, "b =",b, "a == b is:", a == b)')
    a = 5
    b = 7
    print("a =", a, "b =", b, "a == b is:", a == b)
    print(Fore.CYAN + 'Comparison Of Int And Float\n' + Fore.LIGHTGREEN_EX + '>>a = 10\n>>b = 10.0\n'
          + '>>print("a =",a, "b =",b, "a == b is:", a == b)')
    a = 10
    b = 10.0
    print("a =", a, "b =", b, "a == b is:", a == b)
    print(Fore.CYAN + 'Comparison Of Complex Numbers\n' + Fore.LIGHTGREEN_EX + '>>a = 10 + 1j\n>>b = 10. - 1j\n'
          + '>>print("a =",a, "b =",b, "a == b is:", a == b)')
    a = 10 + 1j
    b = 10. - 1j
    print("a =", a, "b =", b, "a == b is:", a == b)
    print(Fore.CYAN + 'Comparison of Booleans\n' + Fore.LIGHTGREEN_EX + '>>a = True\n>>b = False\n'
          + '>>print("a =",a, "b =",b, "a == b is:", a == b)')
    a = True
    b = False
    print("a =", a, "b =", b, "a == b is:", a == b)
    print(Fore.CYAN + 'Comparison Of Different Sequence Types\n' + Fore.LIGHTGREEN_EX + '>>a = (1, 2, 3)\n'
          + '>>b = [1, 2, 3]\n>>print("a =",a, "b =",b, "a == b is:", a == b)')
    a = (1, 2, 3)
    b = [1, 2, 3]
    print("a =", a, "b =", b, "a == b is:", a == b)
    print(Fore.CYAN + 'Comparison Of Strings\n' + Fore.LIGHTGREEN_EX + '>>a = \'BAT\'\n>>b = \'BALL\'\n'
          + '>>print("a =",a, "b =",b, "a == b is:", a == b)')
    a = 'BAT'
    b = 'BALL'
    print("a =", a, "b =", b, "a == b is:", a == b)
    print(Fore.CYAN + 'Comparison Of Tuples\n' + Fore.LIGHTGREEN_EX + '>>a = (1, 2, 4)\n>>b = (1, 2, 3)\n'
          + '>>print("a =",a, "b =",b, "a == b is:", a == b)')
    a = (1, 2, 4)
    b = (1, 2, 3)
    print("a =", a, "b =", b, "a == b is:", a == b)
    print(Fore.CYAN + 'Comparison Of Dictionary Objects\n' + Fore.LIGHTGREEN_EX + '>>a = {1:1, 2:2}\n'
          + '>>b = {2:2, 1:1, 3:3}\n>>print("a =",a, "b =",b, "a == b is:", a == b)')
    a = {1: 1, 2: 2}
    b = {2: 2, 1: 1, 3: 3}
    print("a =", a, "b =", b, "a == b is:", a == b)


def describe_greater_than():
    print(Fore.YELLOW + 'Greater Than Test'
          + '\n\tFrom https://www.tutorialspoint.com/python/python_comparison_operators.htm'
          + Fore.CYAN + '\nBoth Operands Are Integers\n' + Fore.LIGHTGREEN_EX + '>>a = 5\n>>b = 7\n'
          + '>>print("a =",a, "b =",b, "a > b is:", a > b)')
    a = 5
    b = 7
    print("a =", a, "b =", b, "a > b is:", a > b)
    print(Fore.CYAN + 'Comparison Of Int And Float\n' + Fore.LIGHTGREEN_EX + '>>a = 10\n>>b = 10.0\n'
          + '>>print("a =", a, "b =", b, "a > b is:", a > b)')
    a = 10
    b = 10.0
    print("a =", a, "b =", b, "a > b is:", a > b)
    print(Fore.CYAN + 'Comparison Of Complex Numbers\n' + Fore.LIGHTGREEN_EX + '>>a = 10 + 1j\n>>b = 10. - 1j\n'
          + '>>print("a =", a, "b =", b, "a > b is:", a > b)')
    print("TypeError: '>' not supported between instances of 'complex' and 'complex'")
    print(Fore.CYAN + 'Comparison Of Booleans\n' + Fore.LIGHTGREEN_EX + '>>a = True\n>>b = False\n'
          + '>>print("a =", a, "b =", b, "a > b is:", a > b)')
    a = True
    b = False
    print("a =", a, "b =", b, "a > b is:", a > b)
    print(Fore.CYAN + 'Comparison Of Different Sequence Types\n' + Fore.LIGHTGREEN_EX + '>>a = (1, 2, 3)\n'
          + '>>b = [1, 2, 3]>>print("a =", a, "b =", b, "a > b is:", a > b)')
    print("TypeError: '>' not supported between instances of 'tuple' and 'list'")
    print(Fore.CYAN + 'Comparison Of Strings\n' + Fore.LIGHTGREEN_EX + '>>a = \'BAT\'\n>>b = \'BALL\'\n'
          + '>>print("a =", a, "b =", b, "a > b is:", a > b)')
    a = 'BAT'
    b = 'BALL'
    print("a =", a, "b =", b, "a > b is:", a > b)
    print(Fore.CYAN + 'Comparison Of Tuples\n' + Fore.LIGHTGREEN_EX + '>>a = (1, 2, 4)\n>>b = (1, 2, 3)\n'
          + '>>print("a =", a, "b =", b, "a > b is:", a > b)')
    a = (1, 2, 4)
    b = (1, 2, 3)
    print("a =", a, "b =", b, "a > b is:", a > b)
    print(Fore.CYAN + 'Comparison Of Dictionary Objects\n' + Fore.LIGHTGREEN_EX + '>>a = {1:1, 2:2}\n'
          + '>>b = {2:2, 1:1, 3:3}\n>>print("a =", a, "b =", b, "a > b is:", a > b)')
    print("TypeError: '>' not supported between instances of 'dict' and 'dict'")


def describe_greater_or_equal():
    print(Fore.YELLOW + 'Greater Than Or Equal Test'
          + '\n\tFrom https://www.tutorialspoint.com/python/python_comparison_operators.htm\n' + Fore.CYAN
          + 'Both Operands Are Integers\n' + Fore.LIGHTGREEN_EX + '>>a = 5\n>>b = 7\n'
          + '>>print("a =",a, "b =",b, "a >= b is:", a >= b)')
    a = 5
    b = 7
    print("a =", a, "b =", b, "a >= b is:", a >= b)
    print(Fore.CYAN + 'Comparison Of Int And Float\n' + Fore.LIGHTGREEN_EX + '>>a = 10\n>>b = 10.0\n'
          + '>>print("a =",a, "b =",b, "a >= b is:", a >= b)')
    a = 10
    b = 10.0
    print("a =", a, "b =", b, "a >= b is:", a >= b)
    print(Fore.CYAN + 'Comparison Of Complex Numbers\n' + Fore.LIGHTGREEN_EX + '>>a = 10 + 1j\n>>b = 10. - 1j\n'
          + '>>print("a =",a, "b =",b, "a >= b is:", a >= b)')
    print("TypeError: '>=' not supported between instances of 'complex' and 'complex'")
    print(Fore.CYAN + 'Comparison of Booleans\n' + Fore.LIGHTGREEN_EX + '>>a = True\n>>b = False\n'
          + '>>print("a =",a, "b =",b, "a >= b is:", a >= b)')
    a = True
    b = False
    print("a =", a, "b =", b, "a >= b is:", a >= b)
    print(Fore.CYAN + 'Comparison Of Different Sequence Types\n' + Fore.LIGHTGREEN_EX + '>>a = (1, 2, 3)\n'
          + '>>b = [1, 2, 3]\n>>print("a =",a, "b =",b, "a >= b is:", a >= b)')
    print("TypeError: '>=' not supported between instances of 'tuple' and 'list'")
    print(Fore.CYAN + 'Comparison Of Strings\n' + Fore.LIGHTGREEN_EX + '>>a = \'BAT\'\n>>b = \'BALL\'\n'
          + '>>print("a =",a, "b =",b, "a >= b is:", a >= b)')
    a = 'BAT'
    b = 'BALL'
    print("a =", a, "b =", b, "a >= b is:", a >= b)
    print(Fore.CYAN + 'Comparison Of Tuples\n' + Fore.LIGHTGREEN_EX + '>>a = (1, 2, 4)\n>>b = (1, 2, 3)\n'
          + '>>print("a =",a, "b =",b, "a >= b is:", a >= b)')
    a = (1, 2, 4)
    b = (1, 2, 3)
    print("a =", a, "b =", b, "a >= b is:", a >= b)
    print(Fore.CYAN + 'Comparison Of Dictionary Objects\n' + Fore.LIGHTGREEN_EX + '>>a = {1:1, 2:2}\n'
          + '>>b = {2:2, 1:1, 3:3}\n>>print("a =",a, "b =",b, "a >= b is:", a >= b)')
    print("TypeError: '>=' not supported between instances of 'dict' and 'dict'")


def describe_class_method():
    print(Fore.YELLOW + 'Class Method Test\n\tFrom https://www.geeksforgeeks.org/classmethod-in-python/\n'
          + Fore.LIGHTGREEN_EX + '>>class geeks:\n\tcourse = \'DSA\'\n\n\tdef purchase(obj):\n\t\t'
          + 'print("Purchase course : ", obj.course)\n>>geeks.purchase = classmethod(geeks.purchase)\n'
          + '>>geeks.purchase()')
    geeks.purchase = classmethod(geeks.purchase)
    geeks.purchase()
    print(Fore.LIGHTGREEN_EX + '>>class Method_Student:\n\t# create a variable\n\tname = "Geeksforgeeks"\n\n\t'
          + '# create a function\n\tdef print_name(obj):\n\t\tprint("The name is : ", obj.name)\n'
          + '# create print_name class method\n# before creating this line print_name()\n'
          + '# It can be called only with object not with class\n'
          + '>>Method_Student.print_name = classmethod(Method_Student.print_name)\n'
          + '# now this method can be called as classmethod\n# print_name() method is called a class method\n'
          + '>>Method_Student.print_name()')
    Method_Student.print_name = classmethod(Method_Student.print_name)
    Method_Student.print_name()
    print(Fore.LIGHTGREEN_EX + '# Python program to demonstrate\n# use of a class and static method\n'
          + '>>from datetime import date\n\n>>class Person:\n\tdef __init__(self, name, age):\n\t\tself.name = name\n\t'
          + '\tself.age = age\n\n\t# a class method to create a\n\t# Person object by birth year.\n\t@classmethod\n\t'
          + 'def fromBirthYear(cls, name, year):\n\t\treturn cls(name, date.today().year - year)\n\n\t'
          + 'def display(self):\n\t\tprint("Name : ", self.name, "Age : ", self.age)\n\n'
          + '>>person = Person(\'mayank\', 21)\n>>person.display()')
    person = Person('mayank', 21)
    person.display()
    print(Fore.LIGHTGREEN_EX + '>>class Random_Person:\n\tdef __init__(self, name, age):\n\t\tself.name = name\n\t\t'
          + 'self.age = age\n\n\t@staticmethod\n\tdef from_FathersAge(name, fatherAge, fatherPersonAgeDiff):\n\t\t'
          + 'return Random_Person(name, date.today().year - fatherAge + fatherPersonAgeDiff)\n\n\t@classmethod\n\t'
          + 'def from_BirthYear(cls, name, birthYear):\n\t\treturn cls(name, date.today().year - birthYear)\n\n\t'
          + 'def display(self):\n\t\tprint(self.name + "\'s age is: " + str(self.age))\n>>class Man(Random_Person):\n\t'
          + 'sex = \'Female\'\n>>man = Man.from_BirthYear(\'John\', 1985)\n>>print(isinstance(man, Man))')
    man = Man.from_BirthYear('John', 1985)
    print(isinstance(man, Man))
    print(Fore.LIGHTGREEN_EX + '>>man1 = Man.from_FathersAge(\'John\', 1965, 20)\n>>print(isinstance(man1, Man))')
    man1 = Man.from_FathersAge('John', 1965, 20)
    print(isinstance(man1, Man))
    print(Fore.LIGHTGREEN_EX + '# Python program to demonstrate\n# use of a class method and static method.\n'
          + '>>class Old_Person:\n\tdef __init__(self, name, age):\n\t\tself.name = name\n\t\tself.age = age\n\n\t'
          + '# a class method to create a\n\t# Person object by birth year.\n\t@classmethod\n\t'
          + 'def fromBirthYear(cls, name, year):\n\t\t return cls(name, date.today().year - year)\n\n\t'
          + '# a static method to check if a\n\t# Person is adult or not.\n\t@staticmethod\n\tdef isAdult(age):\n\t\t'
          + 'return age > 18\n>>person1 = Old_Person(\'mayank\', 21)\n'
          + '>>person2 = Old_Person.fromBirthYear(\'mayank\', 1996)\n'
          + '>>print(person1.age)')
    person1 = Old_Person('mayank', 21)
    person2 = Old_Person.fromBirthYear('mayank', 1996)
    print(person1.age)
    print(Fore.LIGHTGREEN_EX + '>>print(person2.age)')
    print(person2.age)
    print(Fore.LIGHTGREEN_EX + '>>print(Old_Person.isAdult(22)\n' + Fore.WHITE + f'{Old_Person.isAdult(22)}')


def describe_static_method():
    print(Fore.YELLOW + 'Static Method Test\n\tFrom https://www.geeksforgeeks.org/class-method-vs-static-method-python/'
          + Fore.LIGHTGREEN_EX + '\n>>class Static_Method_Test:\n\tdef __init__(self, value):\n\t\tself.value = value'
          + '\n\n\t@staticmethod\n\tdef get_max_value(x, y):\n\t\treturn max(x, y)\n\n'
          + '# Create an instance of Static_Method_Test\n>>obj = Static_Method_Test(10)\n'
          + '>>print(Static_Method_Test.get_max_value(20, 30))')
    obj = Static_Method_Test(10)
    print(f'{Static_Method_Test.get_max_value(20, 30)}\n' + Fore.LIGHTGREEN_EX + '>>print(obj.get_max_value(20, 30))\n'
          + Fore.WHITE + f'{obj.get_max_value(20, 30)}')


def describe_debug():
    print(Fore.YELLOW + 'Debug Test\n' + Fore.LIGHTGREEN_EX + '# You need to include the -O before the name of your\n'
          + '# file to get this code to work.\n>py -O\n>>if __debug__:\n\tprint(\'Debugging\')\nelse:\n\t'
          + 'print(\'Not Debugging\')')
    if __debug__:
        print('Debugging')
    else:
        print('Not Debugging')


def describe_exporter():
    print(Fore.YELLOW + 'Exporter Test\n' + Fore.LIGHTGREEN_EX + '>>import array\n'
          + '>>a = array.array(\'I\', [1, 2, 3, 4, 5])\n>>b = array.array(\'d\', [1.0, 2.0, 3.0, 4.0, 5.0])\n'
          + '>>c = array.array(\'b\', [5, 3, 1])\n>>x = memoryview(a)\n>>y = memoryview(b)\n>>x == a == y == b')
    a = array.array('I', [1, 2, 3, 4, 5])
    b = array.array('d', [1.0, 2.0, 3.0, 4.0, 5.0])
    c = array.array('b', [5, 3, 1])
    x = memoryview(a)
    y = memoryview(b)
    print(f'{x == a == y == b}\n' + Fore.LIGHTGREEN_EX + '>>x.tolist() == a.tolist() == y.tolist() == b.tolist()\n'
          + Fore.WHITE + f'{x.tolist() == a.tolist() == y.tolist() == b.tolist()}\n' + Fore.LIGHTGREEN_EX
          + '>>z = y[::-2]\n>>z == c')
    z = y[::-2]
    print(f'{z == c}\n' + Fore.LIGHTGREEN_EX + '>>z.tolist() == c.tolist()\n' + Fore.WHITE
          + f'{z.tolist() == c.tolist()}')
    print(Fore.LIGHTGREEN_EX + '>>from ctypes import BigEndianStructure, c_long\n>>class BEPoint(BigEndianStructure):'
          + '\n\t_fields_ = [("x", c_long), ("y", c_long)]\n\n>>point = BEPoint(100, 200)\n>>a = memoryview(point)\n'
          + '>>b = memoryview(point)\n>>a == point')
    point = BEPoint(100, 200)
    a = memoryview(point)
    b = memoryview(point)
    print(f'{a == point}\n' + Fore.LIGHTGREEN_EX + '>>a == b\n' + Fore.WHITE + f'{a == b}\n' + Fore.LIGHTGREEN_EX
          + '>>x is y\n' + Fore.WHITE + f'{x is y}\n' + Fore.LIGHTGREEN_EX + '>>x == y\n' + Fore.WHITE + f'{x == y}')


def describe_import():
    print(Fore.YELLOW + 'Import Test\n' + Fore.LIGHTGREEN_EX + 'spam = __import__(\'spam\', globals(), locals(), [], 0)'
          + 'is the same as import spam\nspam = __import__ (\'spam.ham\', globals(), locals(), [], 0) is the same as'
          + 'import spam.ham\n _temp = __import__(\'spam.ham\', globals(), locals(), [\'eggs\', \'sausage\'], 0)\n '
          + 'eggs = _temp.eggs\n saus = _temp.sausage\n\tis the same as from spam.ham import eggs, sausage as saus')


# Function to calculate speed
def cal_speed(dist, time):
    print(" Distance(km) :", dist)
    print(" Time(hr) :", time)
    return dist / time


# Function to calculate distance traveled
def cal_dis(speed, time):
    print(" Time(hr) :", time)
    print(" Speed(km / hr) :", speed)
    return speed * time


# Function to calculate time taken
def cal_time(dist, speed):
    print(" Distance(km) :", dist)
    print(" Speed(km / hr) :", speed)
    return speed * dist


def describe_absolute_value():
    print(Fore.YELLOW + 'Absolute Value Test\n\tFrom '
          + 'https://www.geeksforgeeks.org/__invert__-and-__abs__-magic-functions-in-python-oops/ and '
          + '\n\thttps://www.geeksforgeeks.org/abs-in-python/\n' + Fore.LIGHTGREEN_EX + '# An integer\n>>var = -94\n'
          + '>>print(\'Absolute value of integer is:\', abs(var))')
    var = -94
    print('Absolute value of integer is:', abs(var))
    print(Fore.LIGHTGREEN_EX + '# floating point number\n>>float_number = -54.26\n'
          + '>>print(\'Absolute value of float is:\', abs(float_number))')
    float_number = -54.26
    print('Absolute value of float is:', abs(float_number))
    print(Fore.LIGHTGREEN_EX + '# A complex number\n>>complex_number = (3 - 4j)\n>>print(\'Absolute value or Magnitude'
          + ' of complex is:\', abs(complex_number))')
    complex_number = (3 - 4j)
    print('Absolute value or Magnitude of complex is:', abs(complex_number))
    print(Fore.LIGHTGREEN_EX + '# Function to calculate speed\n>>def cal_speed(dist, time):\n\tprint(" Distance(km) :'
          + ', dist)\n\tprint(" Time(hr) :", time)\n\treturn dist / time\n\n# Function to calculate distance traveled\n'
          + 'def cal_dis(speed, time):\n\tprint(" Time(hr) :", time)\n\tprint(" Speed(km / hr) :", speed)\n\treturn '
          + 'speed * time\n\n# Function to calculate time taken\ndef cal_time(dist, speed):\n\tprint(" Distance(km) :"'
          + ', dist)\n\tprint(" Speed(km / hr) :", speed)\n\treturn speed * dist\n\n# Driver Code\n# Calling function '
          + 'cal_speed()\n>>print(" The calculated Speed(km / hr) is :", cal_speed(abs(45.9), abs(-2)), "\\n")')
    # Driver Code
    # Calling function cal_speed()
    print(" The calculated Speed(km / hr) is :", cal_speed(abs(45.9), abs(-2)), "\n")
    print(Fore.LIGHTGREEN_EX + '# Calling function cal_dis()\n>>print(" The calculated Distance(km) :", cal_dis('
          + 'abs(-62.9), abs(2.5)), "\\n")')
    print(" The calculated Distance(km) :", cal_dis(abs(-62.9), abs(2.5)), "\n")
    print(Fore.LIGHTGREEN_EX + '# Calling function cal_time()\n'
          + '>>print(" The calculated Time(hr) :", cal_time(abs(48.0), abs(4.5)), "\\n")')
    print(" The calculated Time(hr) :", cal_time(abs(48.0), abs(4.5)), "\n")
    print(Fore.LIGHTGREEN_EX + '>>class ComplexNumber:\n\tdef __init__(self, real, imag):\n\t\tself.real = real\n\t\t'
          + 'self.imag = imag\n\n\tdef __abs__(self):\n\t# Custom implementation for absolute value computation\n\t'
          + 'return (self.real**2 + self.imag**2)**0.5\n\n# Example usage:\n>>complex_num = ComplexNumber(3, 4)\n'
          + '>>absolute_value = abs(complex_num)\n>>print(absolute_value)')
    # Example usage:
    complex_num = ComplexNumber(3, 4)
    absolute_value = abs(complex_num)
    print(absolute_value)
    print(Fore.LIGHTGREEN_EX + '>>class Vector:\n\tdef __init__(self, components):\n\t\tself.components = components'
          + '\n\n\tdef __abs__(self):\n\t\t# Custom implementation for absolute value computation\n\t\t'
          + 'return sum(component**2 for component in self.components)**0.5\n\n# Example usage:\n'
          + '>>vector = Vector([1, 2, 3])\n>>magnitude = abs(vector)\nprint(magnitude)')
    # Example usage:
    vector = Vector([1, 2, 3])
    magnitude = abs(vector)
    print(magnitude)


def describe_add():
    print(Fore.YELLOW + 'Add Test\n\tFrom https://www.w3schools.com/python/ref_set_add.asp\n' + Fore.LIGHTGREEN_EX
          + '>>fruits = {"apple", "banana", "cherry"}\n>>fruits.add("orange")\n>>print(fruits)')
    fruits = {"apple", "banana", "cherry"}
    fruits.add("orange")
    print(fruits)
    print(Fore.LIGHTGREEN_EX + '>>fruits = {"grapes", "lemons", "strawberries"}\n>>fruits.add("strawberries")\n'
          + '>>print(fruits)')
    fruits = {"grapes", "lemons", "strawberries"}
    fruits.add("strawberries")
    print(fruits)


async def async_generator():
    for i in range(5):
        yield i


async def async_iterator():
    for i in range(3):
        yield i


async def custom_async_iterator():
    for i in range(2):
        yield i


async def control():
    async_iter = aiter(async_generator())
    async for item in async_iter:
        print(item, end=' ')
    print('')
    aiter_obj = aiter(async_iterator())
    async for stuff in aiter_obj:
        print(stuff, end=' ')
    print('')
    async_iterable = aiter(custom_async_iterator())
    async for individual in async_iterable:
        print(individual, end=' ')
    print('')


def describe_aiter():
    print(Fore.YELLOW + 'Asynchronous Iterable Test\n\tFrom https://diveintopython.org/functions/built-in/aiter\n'
          + Fore.LIGHTGREEN_EX + '>>from asyncio import run\n'
          + '>>async def async_generator():\n\tfor i in range(5):\n\t\tyield i\n\n'
          + '>>async def async_iterator():\n\tfor i in range(3):\n\t\tyield i\n\n'
          + '>>async def custom_async_iterator():\n\tfor i in range(2):\n\t\tyield i\n\n'
          + '>>async def control():\n\tasync_iter = aiter(async_generator())\n\tasync for item in async_iter:\n\t\t'
          + 'print(item, end=\' \')\n\tprint(\'\')\n\t'
          + 'aiter_obj = aiter(async_iterator())\n\tasync for stuff in aiter_obj:\n\t\t'
          + 'print(stuff, end=\' \')\n\tprint(\'\')\n\tasync_iterable = aiter(custom_async_iterator())\n\t'
          + 'async for individual in async_iterable:\n\t\tprint(individual, end=\' \')\n\tprint(\'\')\n'
          + '>>run(control())')
    run(control())


def describe_all():
    print(Fore.YELLOW + 'All Test\n\tFrom https://www.w3schools.com/python/ref_func_all.asp\n' + Fore.LIGHTGREEN_EX
          + '>>mylist = [True, True, True]\n>>x = all(mylist)\n>>print(x)')
    mylist = [True, True, True]
    x = all(mylist)
    print(f'{x}\n' + Fore.LIGHTGREEN_EX + '>>mylist = [0, 1, 1]\n>>x = all(mylist)\n>>print(x)')
    mylist = [0, 1, 1]
    x = all(mylist)
    print(f'{x}\n' + Fore.LIGHTGREEN_EX + '>>mytuple = (0, True, False)\n>>x = all(mytuple)\n>>print(x)')
    mytuple = (0, True, False)
    x = all(mytuple)
    print(f'{x}\n' + Fore.LIGHTGREEN_EX + '>>myset = {0, 1, 0}\n>>x = all(myset)\n>>print(x)')
    myset = {0, 1, 0}
    x = all(myset)
    print(f'{x}\n' + Fore.LIGHTGREEN_EX + '>>mydict = {0 : "Apple", 1 : "Orange"}\nx = all(mydict)\n>>print(x)')
    mydict = {0: "Apple", 1: "Orange"}
    x = all(mydict)
    print(f'{x}\n' + Fore.CYAN + 'Note all() is equivalent to:\n\tdef all(iterable):\n\t\tfor element in iterable:'
          + '\n\t\t\tif not element:\n\t\t\t\treturn False\n\t\treturn True')


def describe_any():
    print(Fore.YELLOW + 'Any Test\n\tFrom https://www.w3schools.com/python/ref_func_any.asp\n' + Fore.LIGHTGREEN_EX
          + '>>mylist = [False, True, False]\n>>x = any(mylist)\nprint(x)')
    mylist = [False, True, False]
    x = any(mylist)
    print(f'{x}\n' + Fore.LIGHTGREEN_EX + '>>mytuple = (0, 1, False)\n>>x = any(mytuple)\n>>print(x)')
    mytuple = (0, 1, False)
    x = any(mytuple)
    print(f'{x}\n' + Fore.LIGHTGREEN_EX + '>>myset = {0, 1, 0}\n>>x = any(myset)\n>>print(x)')
    myset = {0, 1, 0}
    x = any(myset)
    print(f'{x}\n' + Fore.LIGHTGREEN_EX + '>>mydict = {0: "Apple", 1: "Orange"}\n>>x = any(mydict)\n>>print(x)')
    mydict = {0: "Apple", 1: "Orange"}
    x = any(mydict)
    print(f'{x}\n' + Fore.CYAN + 'Note any() is equivalent to:\n\tdef any(iterable):\n\t\tfor element in iterable:'
          + '\n\t\t\tif element:\n\t\t\t\treturn True\n\t\treturn False')


def describe_ascii():
    print(Fore.YELLOW + 'Ascii Test\n\tFrom '
          + 'https://www.w3schools.com/python/ref_func_ascii.asp#:~:text=The%20ascii()%20function%20returns,will%20be%'
          + '\n\t20replaced%20with%20%5Cxe5%20.\n' + Fore.LIGHTGREEN_EX + '>>x = ascii("My name is Ståle")\n'
          + '>>print(x)')
    x = ascii("My name is Ståle")
    print(x)
    print(Fore.LIGHTGREEN_EX + '>>x = ascii("My n\u05d4me is Ståle")\n>>print(x)')
    x = ascii("My n\u05d4me is Ståle")
    print(x)
    print(Fore.LIGHTGREEN_EX + '>>x = ascii("My n\u0002m\U000002f9 is St\x21le")>>print(x)')
    x = ascii("My n\u0002m\U000002f9 is St\xe6le")
    print(x)


async def Odd_Control():
    async for c in AsyncOddCounter(100):
        print(c)


async def all_keys(key):
    keys = {"key1": 1234, "key2": 2345,
            "key3": 3456, "key4": 4567,
            "key5": 5678}
    return keys.get(key)


async def Key_Control():
    async for c in KeyTaker(["key1", "key2", "key3"]):
        print(c)


def describe_asynchronous_iterable_iterator():
    print(Fore.YELLOW + 'Asynchronous Iterable Test\n\tFrom https://medium.com/geekculture/asynchronous-iterators-in-'
          + 'python-fdf55198287d\n' + Fore.CYAN + 'Example of an Iterator:\n\t' + '\u2022   '
          + 'An iterator must implement the __iter__ special method.\n\t' + '\u2022   '
          + 'The __iter__ special method should return an iterable.i.e. Any object\n\t'
          + '    that implements the __next__ special method. This could be its own\n\t    '
          + 'class(self) or any other class object.\n\t\u2022   '
          + 'The __next__ method has the logic to run the iterator until a condition\n\t    is satisfied.'
          + '\n\t\u2022   Once the condition is satisfied, StopIteration error message should be\n\t    raised.\n'
          + Fore.LIGHTGREEN_EX
          + '>>class OddCounter:\n\tdef __init__(self, end_range):\n\t\tself.start = -1\n\t\tif not end_range:\n\t\t\t'
          + 'raise ValueError("end_range value should be specified")\n\t\tself.end = end_range\n\n\tdef __iter__(self):'
          + '\n\t\treturn self\n\n\tdef __next__(self):\n\t\tif self.start < self.end-1:\n\t\t\tself.start += 2\n\t\t\t'
          + 'return self.start\n\t\telse:\n\t\t\traise StopIteration\n\n# iterate over oddcounter with range 100\n'
          + 'for c in OddCounter(100):\n\tprint(c)')
    # iterate over oddcounter with range 100
    for c in OddCounter(100):
        print(c)
    print(Fore.CYAN + 'Example of Async Iterator:\n\tAn async iterator typically contains,\n\t\u2022   '
          + 'A __aiter__() method instead of __iter__() method.\n\t\u2022   '
          + 'The __aiter()__ method must return an object that implements a async\n\t    def __anext__().\n\t\u2022   '
          + 'The __anext__() method must return a value for every single iteration\n\t    '
          + 'and raise StopAsyncIteration at the end instead of StopIteration.\n' + Fore.LIGHTGREEN_EX
          + '>>class AsyncOddCounter:\n\tdef __init__(self, end_range):\n\t\tself.end = end_range\n\t\tself.start = -1'
          + '\n\n\tdef __aiter__(self):\n\t\treturn self\n\tasync def __anext__(self):\n\t\tif self.start < self.end-1:'
          + '\n\t\t\tself.start += 2\n\t\t\treturn self.start\n\t\telse:\n\t\t\traise StopAsync Iteration\n\n'
          + '>>async def Odd_Control():\n\tasync for c in OddCounter(100):\n\t\tprint(c)\n\n'
          + '>>run(Odd_Control())')
    run(Odd_Control())
    print(Fore.LIGHTGREEN_EX + '>>async def all_keys(key):\n\tkeys = {"key1": 1234, "key2": 2345,\n\t\t"key3": 3456,'
          + ' "key4": 4567, "key5": 5678}\n\treturn keys.get(key)\n'
          + '>>class KeyTaker:\n\tdef __init__(self, keys):\n\t\tself.keys = keys\n\n\tdef __aiter__(self):\n\t\t'
          + '# create an iterator of the input keys\n\t\tself.iter_keys = iter(self.keys)\n\t\treturn self\n\n\t'
          + 'async def __anext__(self):\n\t\ttry:\n\t\t\t# extract the keys one at a time\n\t\t\t'
          + 'k = next(self.iter_keys)\n\t\texcept StopIteration:\n\t\t\t# raise stopasynciteration at the end of '
          + 'iterator\n\t\t\traise StopAsyncIteration\n\t\t# return values for a key\n\t\tvalue = await all_keys(k)\n'
          + '\t\treturn value\n\n>>async def Key_Control():\n\tasync for c in KeyTaker(["key1", "key2", "key3"]):\n\t\t'
          + 'print(c)\n\n>>run(Key_Control())')
    run(Key_Control())


async def nested():
    return 42


async def nested_control():
    nested()
    # RuntimeWarning: Enable tracemalloc to get the object allocation traceback 42
    print(await nested())


async def nested_control_two():
    task = create_task(nested())
    await task


async def nested_control_three():
    # This is what I believe a future is supposed to look like.
    await nested()
    await gather(
        nested(),
        nested_control()
    )


def describe_awaitable():
    print(Fore.YELLOW + 'Awaitable Test\n\tFrom The Python Standard Library -> Networking and Interprocess '
          + 'Communication\n\t -> asyncio -- Asynchronous I/O -> Coroutines and Tasks\n' + Fore.CYAN + 'Coroutines\n'
          + Fore.LIGHTGREEN_EX + '>>import asyncio\n>>async def nested():\n\treturn 42\n\n>>async def nested_control():'
          + '\n\t# Nothing happens if we just call "nested()".\n\t# A coroutine object is created but not awaited,\n\t'
          + '# so it *won\'t run at all*.\n\tnested()\n\n\t# Let\'s do it differently now and await it:\n\t'
          + 'print(await nested()) # will print "42".\n\n>>run(nested_control())')
    run(nested_control())
    print(Fore.CYAN
          + 'Important In this documentation the term “coroutine” can be used for two closely related concepts:'
          + '\n\t\u2022   a coroutine function: an async def function;'
          + '\n\t\u2022   a coroutine object: an object returned by calling a coroutine function.\n'
          + 'Tasks\n\tTasks are used to schedule coroutines concurrently.\n' + Fore.LIGHTGREEN_EX
          + '>>async def nested_control_two():\n\t# Schedule nested() to run soon concurrently\n\t# with'
          + '"nested_control_two():"\n\ttask = create_task(nested())\n\n\t# "task" can now be used to cancel "nested()"'
          + ', or\n\t# can simply be awaited to wait until it is complete:\n\tawait task\n\n>>run(nested_control_two())'
          )
    run(nested_control_two())
    print(Fore.CYAN + 'Future\n\tA Future is a special low-level awaitable object that represents an eventual\n\t'
          + 'result of an asynchronous operation.\nasync def nested_control_three():\n\t'
          + 'await function_that_returns_a_future_object()\n\n\t# this is also valid:\n\tawait asyncio.gather(\n\t\t'
          + 'function_that_returns_a_future_object(),\n\t\tsome_python_coroutine()\n\t)\n' + Fore.LIGHTGREEN_EX
          + 'async def nested_control_three():\n\t# This is what I believe a future is supposed to look like.'
          + '\n\tawait nested()\n\tawait gather(\n\t\tnested(),\n\t\tnested_control()\n\t)\n'
          + '>>run(nested_control_three)')
    run(nested_control_three())


def describe_awaitable_anext():
    print(Fore.YELLOW + 'Awaitable Anext Test\n' + Fore.LIGHTGREEN_EX + '>>class KeyTaker:\n\tdef __init__(self, keys):'
          + '\n\t\tself.keys = keys\n\n\tdef __aiter__(self):\n\t\t# create an iterator of the input keys\n\t\t'
          + 'self.iter_keys = iter(self.keys)\n\t\treturn self\n\n\t'
          + 'async def __anext__(self): # This is the part of the function we are examining.\n\t\ttry:\n\t\t\t'
          + '# extract the keys one at a time\n\t\t\tk = next(self.iter_keys) # The next item in the iterator.'
          + '\n\t\texcept StopIteration:\n\t\t\t'
          + '# raise stopasynciteration at the end of iterator\n\t\t\traise StopAsyncIteration\n\t\t'
          + '# return values for a key\n\t\tvalue = await all_keys(k) # The Awaitable\n\t\treturn value')


def describe_bin():
    print(Fore.YELLOW + 'Bin Test\n' + Fore.LIGHTGREEN_EX + '>>a = bin(3)>>print(a)')
    a = bin(3)
    print(f'{a}\n' + Fore.LIGHTGREEN_EX + '>>b = bin(-10)\n>>print(b)')
    b = bin(-10)
    print(f'{b}\n' + Fore.LIGHTGREEN_EX + '>>c = format(14, \'#b\'), format(14, \'b\')\n>>print(c)')
    c = format(14, '#b'), format(14, 'b')
    print(f'{c}\n' + Fore.LIGHTGREEN_EX + '>>d = f\'{14:#b}\', f\'{14:b}\'\n>>print(d)')
    d = f'{14:#b}', f'{14:b}'
    print(d)


def myFunction():
    return True


def describe_boolean_values():
    print(Fore.YELLOW + 'Boolean Values Test\n' + Fore.LIGHTGREEN_EX + '>>print(10 > 9)\n' + Fore.WHITE
          + '{0}\n'.format(10 > 9) + Fore.LIGHTGREEN_EX + 'print(10 == 9)\n' + Fore.WHITE + '{0}\n'.format(10 == 9)
          + Fore.LIGHTGREEN_EX + '>>print(10 < 9)\n' + Fore.WHITE + '{0}\n'.format(10 < 9) + Fore.LIGHTGREEN_EX
          + '>>a = 200\n>>b = 33\n>>if b < a:\n\tprint("b is greater than a")\nelse:\n\t'
          + 'print("b is not greater than a")')
    a = 200
    b = 33
    if b > a:
        print("b is greater than a")
    else:
        print("b is not greater than a")
    print(Fore.LIGHTGREEN_EX + '>>print(bool("Hello"))\n' + Fore.WHITE + '{0}\n'.format(bool("Hello"))
          + Fore.LIGHTGREEN_EX + '>>print(bool(15))\n' + Fore.WHITE + '{0}\n'.format(bool(15)) + Fore.LIGHTGREEN_EX
          + '>>x = "Hello"\n>>y = 15\n>>print(bool(x))')
    x = "Hello"
    y = 15
    print('{0}\n'.format(bool(x)) + Fore.LIGHTGREEN_EX + '>>print(bool(y))\n' + Fore.WHITE + '{0}\n'.format(bool(y))
          + Fore.LIGHTGREEN_EX + '>>bool("abc")\n' + Fore.WHITE + '{0}\n'.format(bool("abc")) + Fore.LIGHTGREEN_EX
          + '>>bool(123)\n' + Fore.WHITE + '{0}\n'.format(bool(123)) + Fore.LIGHTGREEN_EX
          + '>>bool(["apple", "cherry", "banana"])\n' + Fore.WHITE + '{0}\n'.format(bool(["apple", "cherry", "banana"]))
          + Fore.LIGHTGREEN_EX + '>>bool(False)\n' + Fore.WHITE + '{0}\n'.format(bool(False)) + Fore.LIGHTGREEN_EX
          + '>>bool(None)\n' + Fore.WHITE + '{0}\n'.format(bool(None)) + Fore.LIGHTGREEN_EX + '>>bool(0)\n' + Fore.WHITE
          + '{0}\n'.format(bool(0)) + Fore.LIGHTGREEN_EX + '>>bool("")\n' + Fore.WHITE + '{0}\n'.format(bool(""))
          + Fore.LIGHTGREEN_EX + '>>bool(())\n' + Fore.WHITE + '{0}\n'.format(bool(())) + Fore.LIGHTGREEN_EX
          + '>>bool([])\n' + Fore.WHITE + '{0}\n'.format(bool([])) + Fore.LIGHTGREEN_EX + '>>bool({})\n' + Fore.WHITE
          + '{0}\n'.format(bool({})) + Fore.LIGHTGREEN_EX + '>>class Bool_My_Class():\n\tdef __len__(self):\n\t\t'
          + 'return 0\n>>myobj = Bool_My_Class()\n>>print(bool(myobj)')
    myobj = Bool_My_Class()
    print('{0}\n'.format(bool(myobj)) + Fore.LIGHTGREEN_EX + '>>def myFunction():\n\treturn True\n'
          + '>>print(myFunction())\n' + Fore.WHITE + '{0}\n'.format(myFunction()) + Fore.LIGHTGREEN_EX
          + '>>if myFunction():\n\tprint("YES!")\nelse:\n\tprint("NO!")')
    if myFunction():
        print("YES!")
    else:
        print("NO!")
    print(Fore.LIGHTGREEN_EX + ">>x = 200\n>>print(isinstance(x, int))")
    x = 200
    print(isinstance(x, int))


def describe_breakpoint():
    print(Fore.YELLOW + 'Breakpoint Test\n' + Fore.LIGHTGREEN_EX + '>>x = 10\n>>y = \'Hi\'\n>>z = \'Hello\'\n'
          + '>>print(y)')
    x = 10
    y = 'Hi'
    z = 'Hello'
    print('{0}\n'.format(y) + Fore.LIGHTGREEN_EX + '>>breakpoint()\n>>print(z)\n' + Fore.WHITE
          + '> file path\n -> print(z)\n(Pbd) c\nHello\n$\n' + Fore.CYAN + 'Symbols:\n\tc: continue execution\n\t'
          + 'q: quit the debugger/execution\n\tn: step to next line within the same function\n\t'
          + 's: step to next line in this function or a called function')


def describe_bytes_capitalize():
    print(Fore.YELLOW + 'Bytes Capitalize Test\n' + Fore.LIGHTGREEN_EX
          + '>>txt_bytes = b\'hello, and welcome to my world.\'\n>>x = txt_bytes.capitalize()\n>>print(x)')
    txt_bytes = b'hello, and welcome to my world.'
    x = txt_bytes.capitalize()
    print('{0}\n'.format(x) + Fore.LIGHTGREEN_EX + '>>txt_bytearray = bytearray(b\'python is FUN!\'\n'
          + '>>x = txt_bytearray.capitalize()\n>>print(x)')
    txt_bytearray = bytearray(b'python is FUN!')
    x = txt_bytearray.capitalize()
    print('{0}\n'.format(x) + Fore.LIGHTGREEN_EX + '>>txt_bytearray_number = bytearray(\'36 is my age.\')\n'
          + '>>x = txt_bytearray_number.capitalize()\n>>print(x)')
    txt_bytearray_number = bytearray(b'36 is my age.')
    x = txt_bytearray_number.capitalize()
    print(x)


def describe_bytes_center():
    print(Fore.YELLOW + 'Bytes Centered Test\n' + Fore.LIGHTGREEN_EX + '>>txt_bytes = b\'banana\''
          + '>>x = txt_bytes.center(4)\n>>print(x)')
    txt_bytes = b'banana'
    x = txt_bytes.center(4)
    print('{0}\n'.format(x) + Fore.LIGHTGREEN_EX + '>>txt_bytearray = bytearray(b\'banana\')\n'
          + '>>x = txt_bytearray.center(4)\n>>print(x)')
    txt_bytearray = bytearray(b'banana')
    x = txt_bytearray.center(4)
    print('{0}\n'.format(x) + Fore.LIGHTGREEN_EX + '>>x = txt_bytearray.center(20)\n>>print(x)')
    x = txt_bytearray.center(20)
    print('{0}\n'.format(x) + Fore.LIGHTGREEN_EX + '>>x = txt_bytearray.center(20, b\'S\')\n>>print(x)')
    x = txt_bytearray.center(20, b'S')
    print(x)


def describe_bytes_count():
    print(Fore.YELLOW + 'Bytes Count Test\n' + Fore.LIGHTGREEN_EX + '# Create a bytes object and a bytearray.\n'
          + '# bytes type object\n>>data = bytes(b"aabb bccc")\n# bytearray type object'
          + '\n>>arr = bytearray(b"aab bcccc")\n'
          + '>>print("Count of c between [start, end] or [2, 5]:", data.count(b"c", 2, 5)')
    # Create a bytes object and a bytearray.
    # bytes type object
    data = bytes(b"aabb bccc")
    # bytearray type object
    arr = bytearray(b"aab bcccc")
    print("Count of c between [start, end] or [2, 5]:", data.count(b"c", 2, 5))
    print(Fore.LIGHTGREEN_EX + '>>print("Count of c between [start, end] or [3, 7]:", arr.count(b"c", 3, 7))')
    print("Count of c between [start, end] or [3, 7]:", arr.count(b"c", 3, 7))
    print(Fore.LIGHTGREEN_EX + '>>print("Count of 98 between [start, end] or [2, 5]:", data.count(98, 2, 5)')
    print("Count of 97 between [start, end] or [2, 5]:", data.count(98, 2, 5))
    print(Fore.LIGHTGREEN_EX + '>>print("Count of 98 between [start, end] or [3, 7]:", arr.count(98, 3, 7))')
    print("Count of 98 between [start, end] or [3, 7]:", arr.count(98, 3, 7))
    print(Fore.LIGHTGREEN_EX + '>>print("Count of \'\' between [start, end] or [2, 5]:", data.count(b\'\', 2, 5))')
    print("Count of ' ' between [start, end] or [2, 5]:", data.count(b'', 2, 5))
    print(Fore.LIGHTGREEN_EX + '>>print("Count of \'\' between [start, end] or [3, 7]:", arr.count(b\'\', 3, 7))')
    print("Count of ' ' between [start, end] or [3, 7]:", arr.count(b'', 3, 7))


def describe_bytes_decode():
    print(Fore.YELLOW + 'Bytes Decode Test\n\t'
          + 'From https://www.digitalocean.com/community/tutorials/python-string-encode-decode\n' + Fore.LIGHTGREEN_EX
          + '>>str_original = \'Hello\'\n>>bytes_endcoded = str_original.encode(encoding=\'utf-8\')\n'
          + '>>print(type(bytes_encoded))')
    str_original = 'Hello'
    bytes_encoded = str_original.encode(encoding='utf-8')
    print(str(type(bytes_encoded)) + Fore.LIGHTGREEN_EX
          + '\n>>str_decoded = bytes_encoded.decode()\n>>print(type(str_decoded))')
    str_decoded = bytes_encoded.decode()
    print(str(type(str_decoded)) + Fore.LIGHTGREEN_EX + '\n>>print(\'Encoded bytes=\', bytes_encoded)\n'
          + Fore.WHITE + 'Encoded bytes = {0}\n'.format(bytes_encoded) + Fore.LIGHTGREEN_EX
          + '>>print(\'Decoded String =\', bytes_encoded)\n' + Fore.WHITE + 'Decoded String = {0}\n'.format(str_decoded)
          + Fore.LIGHTGREEN_EX + '>>print(\'str_original equals str_decoded =\', str_original == str_decoded)\n'
          + Fore.WHITE + 'str_original equals str_decoded = {0}\n'.format(str_original == str_decoded)
          + Fore.LIGHTGREEN_EX + '>>bytes_without_encoding = b\'Hello%\\xe2\\x99\\xa3\'\n'
          + '>>bytearray_without_encoding = bytearray(b\'Hello%\\xe2\\x99\\xa3\'\n'
          + '>>bytes_strict = bytes_without_encoding.decode(encoding=\'ascii\', errors=\'strict\')\n'
          + '>>bytearray_strict = bytearray_without_encoding.decode(encoding=\'ascii\', errors=\'strict\')\n'
          + '>>bytes_ignore = bytes_without_encoding.decode(encoding=\'ascii\', errors=\'ignore\')\n'
          + '>>bytearray_ignore = bytearray_without_encoding.decode(encoding=\'ascii\', errors=\'ignore\')\n'
          + '>>bytes_replace = bytes_without_encoding.decode(encoding=\'ascii\', errors=\'replace\')\n'
          + '>>bytearray_replace = bytearray_without_encoding.decode(encoding=\'ascii\', errors=\'replace\')\n'
          + '>>print(\'b\\\'Hello%\\U00000005\\\' decoded with errors=\\\'ignore\\\': {0}\\n\''
          + '.format(bytes_ignore)\n\t+ \'bytearray(b\\\'Hello!\\U00000005\\\') decoded with '
          + 'errors=\\\'ignore\\\': {0}\\n\'.format(bytearray_ignore)\n\t+ \'b\\\'Hello%\\U00000005\\\' decoded with'
          + ' errors=\\\'replace\\\': {0}\\n\'.format(bytes_replace)\n\t+ \'bytearray(b\\\'Hello!\\U00000005\\\')'
          + ' decoded with errors=\\\'replace\\\': {0}\'.format(bytearray_replace))')
    bytes_without_encoding = b'Hello%\xe2\x99\xa3'
    bytearray_without_encoding = bytearray(b'Hello%\xe2\x99\xa3')
    bytes_ignore = bytes_without_encoding.decode(encoding='ascii', errors='ignore')
    bytearray_ignore = bytearray_without_encoding.decode(encoding='ascii', errors='ignore')
    bytes_replace = bytes_without_encoding.decode(encoding='ascii', errors='replace')
    bytearray_replace = bytearray_without_encoding.decode(encoding='ascii', errors='replace')
    print('UnicodeDecodeError: \'ascii\' codec can\'t decode byte 0xe2 in position 6: ordinal not in range(128)\n'
          + 'UnicodeDecodeError: \'ascii\' codec can\'t decode byte 0xe2 in position 6: ordinal not in range(128)\n'
          + 'b\'Hello%\U00000005\' decoded with errors=\'ignore\': {0}\n'.format(bytes_ignore)
          + 'bytearray(b\'Hello!\U00000005\') decoded with errors=\'ignore\': {0}\n'.format(bytearray_ignore)
          + 'b\'Hello%\U00000005\' decoded with errors=\'replace\': {0}\n'.format(bytes_replace)
          + 'bytearray(b\'Hello!\U00000005\') decoded with errors=\'replace\': {0}\n'.format(bytearray_replace)
          + Fore.LIGHTGREEN_EX + '>>str_original = input(\'Please enter string data\\n\'\n'
          + '>>bytes_encoded = str_original.encode()\n>>str_decoded = bytes_encoded.decode()\n'
          + '>>print(\'Encoded bytes =\', bytes_encoded)')
    str_original = 'aåb∫cçd∂e´´´ƒg©1¡'
    print('Please enter string data:\n{0}'.format(str_original))
    bytes_encoded = str_original.encode()
    str_decoded = bytes_encoded.decode()
    print('Encoded bytes = {0}\n'.format(bytes_encoded) + Fore.LIGHTGREEN_EX
          + '>>print(\'Decoded String =\', str_decoded)\n' + Fore.WHITE + 'Decoded String = {0}\n'.format(str_decoded)
          + Fore.LIGHTGREEN_EX + 'print(\'str_original equals str_decoded =\', str_original == str_decoded)\n'
          + Fore.WHITE + 'str_original equals str_decoded = {0}'.format(str_original == str_decoded))


def describe_bytes_endswith():
    print(Fore.YELLOW + 'Bytes Endswith Test\n\tFrom https://www.w3schools.com/python/ref_string_endswith.asp\n'
          + Fore.LIGHTGREEN_EX + '>>txt_byte = b\'Hello, welcome to my world."\n>>x = txt_byte.endswith(b\'.\')\n'
          + '>>print(x)')
    txt_byte = b'Hello, welcome to my world.'
    x = txt_byte.endswith(b'.')
    print('{0}\n'.format(x) + Fore.LIGHTGREEN_EX + '>>x = txt_byte.endswith("my world.")\n>>print(x)')
    x = txt_byte.endswith(b'my world.')
    print('{0}\n'.format(x) + Fore.LIGHTGREEN_EX + '>>x = txt_byte.endswith((b\'welcome\', b\'my\'))\n>>print(x)')
    x = txt_byte.endswith((b'welcome', b'my'))
    print('{0}\n'.format(x) + Fore.LIGHTGREEN_EX + '>>x = txt_byte.endswith((b\'welcome\', b\'my world.\'))'
          + '\n>>print(x)')
    x = txt_byte.endswith((b'welcome', b'my world.'))
    print('{0}\n'.format(x) + Fore.LIGHTGREEN_EX + '>>x = txt_byte.endswith(b\'my world.\', 5, 11)\n>>print(x)')
    x = txt_byte.endswith(b'my world.', 5, 11)
    print('{0}\n'.format(x) + Fore.LIGHTGREEN_EX + '>>x = txt_byte.endswith(b\'welc\', 5, 11)\n>>print(x)')
    x = txt_byte.endswith(b'welc', 5, 11)
    print(x)


def describe_bytes_expandtabs():
    print(Fore.YELLOW + 'Bytes Expand Tabs Test\n' + Fore.LIGHTGREEN_EX + '>>txt = b\'01\\t012\\t0123\\t01234\'\n'
          + '>>print(f\'{txt.expandtabs()}\\n{txt.expandtabs(4)}\')')
    txt = b'01\t012\t0123\t01234'
    print(f'{txt.expandtabs()}\n{txt.expandtabs(4)}')


def describe_bytes_find():
    print(Fore.YELLOW + 'Bytes Find Method Test\n\tFrom https://www.w3schools.com/python/ref_string_find.asp\n'
          + Fore.LIGHTGREEN_EX + '>>txt_bytes = b\'Hello, welcome to my world.\'\n>>x = txt_bytes.find(b\'welcome\')\n'
          + '>>print(x)')
    txt_bytes = b'Hello, welcome to my world.'
    x = txt_bytes.find(b'welcome')
    print('{0}\n'.format(x) + Fore.LIGHTGREEN_EX + '>>x = txt_bytes.find(b\'e\')\n>>print(x)')
    x = txt_bytes.find(b'e')
    print('{0}\n'.format(x) + Fore.LIGHTGREEN_EX + '>>x = txt_bytes.find(b\'e\', 5, 10)\n>>print(x)')
    x = txt_bytes.find(b'e', 5, 10)
    print('{0}\n'.format(x) + Fore.LIGHTGREEN_EX + '>>print(txt_bytes.find(b\'q\'))\n'
          + Fore.WHITE + '{0}\n'.format(txt_bytes.find(b'q')) + Fore.LIGHTGREEN_EX
          + '>>import traceback\n>>try:\n\tprint(txt_bytes.index(b\'q\')\n  except ValueError as e:\n\t'
          + 'print(traceback.format_exc())')
    try:
        print(txt_bytes.index(b'q'))
    except ValueError as e:
        print(traceback.format_exc())


def describe_bytes_index():
    print(Fore.YELLOW + 'Bytes Index Method Test\n\tFrom https://www.w3schools.com/python/ref_list_index.asp\n'
          + Fore.LIGHTGREEN_EX + '>>fruits = [b\'apple\', b\'banana\', b\'cherry\']\n>>x = fruits.index(b\'cherry\')\n'
          + 'print(x)')
    fruits = [b'apple', b'banana', b'cherry']
    x = fruits.index(b'cherry')
    print('{0}\n'.format(x) + Fore.LIGHTGREEN_EX + '>>x = fruits[0].index(b\'p\', 1, 2)\n>>print(x)')
    x = fruits[1].index(b'a', 2, 4)
    print('{0}\n'.format(x) + Fore.LIGHTGREEN_EX + '>>import traceback\n>>try:\n\tprint(fruits[2].index(b\'n\', 2, 5)'
          + '\n  except ValueError as e:\n\tprint(traceback.format_exc())')
    try:
        print(fruits[2].index(b'n', 2, 5))
    except ValueError as e:
        print(traceback.format_exc())


def describe_bytes_isalnum():
    print(Fore.YELLOW + 'Bytes isalnum Method Test\n' + Fore.LIGHTGREEN_EX + '>>print(b\'ABCabc1\'.isalnum())\n'
          + Fore.WHITE + '{0}\n'.format(b'ABCabc1'.isalnum()) + Fore.LIGHTGREEN_EX
          + '>>print(b\'ABC abc1\'.isalnum())\n' + Fore.WHITE + '{0}\n'.format(b'ABC abc1'.isalnum()))


def describe_bytes_isalpha():
    print(Fore.YELLOW + 'Bytes isalpha Method Test\n\tFrom https://www.w3schools.com/python/ref_string_isalpha.asp\n'
          + Fore.LIGHTGREEN_EX + '>>txt = b\'CompanyX\'\n>>x = txt.isalpha()\n>>print(x)')
    txt = b'CompanyX'
    x = txt.isalpha()
    print('{0}\n'.format(x) + Fore.LIGHTGREEN_EX + '>>txt = b\'Company10\'\n>>x = txt.isalpha()\n>>print(x)')
    txt = b'Company10'
    x = txt.isalpha()
    print(x)


def describe_bytes_isascii():
    print(Fore.YELLOW + 'Bytes isascii Method Test\n\tFrom https://www.w3schools.com/python/ref_string_isascii.asp\n'
          + Fore.LIGHTGREEN_EX + '>>print(b\'\'.isascii())\n' + Fore.WHITE + '{0}\n'.format(b''.isascii())
          + Fore.LIGHTGREEN_EX + '>>print(b\'Company123\'.isascii())\n' + Fore.WHITE
          + '{0}\n'.format(b'Company123'.isascii()) + Fore.LIGHTGREEN_EX
          + '>>print(\'\\x09\\xa5\\xd2\\xc3\\xb3\\xc3\\xa3\\x19\\x35\\x67\\x59\\x23\\x43\\x38\\x41\')\n' + Fore.WHITE
          + '\x09\x15\x22\x33\x23\x13\x43\x19\x35\x67\x59\x44\x42\x38\x41\n' + Fore.LIGHTGREEN_EX
          + '>>print(b\'\\x09\\xa5\\xd2\\xc3\\xb3\\xc3\\xa3\\x19\\x35\\x67\\x59\\x23\\x43\\x38\\x41\')\n' + Fore.WHITE
          + '{0}\n'.format(b'\x09\x15\x22\x33\x23\x13\x43\x19\x35\x67\x59\x44\x42\x38\x41') + Fore.LIGHTGREEN_EX
          + '>>print(b\'\\x09\\xa5\\xd2\\xc3\\xb3\\xc3\\xa3\\x19\\x35\\x67\\x59\\x23\\x43\\x38\\x41\'.isascii())\n'
          + Fore.WHITE + '{0}\n'.format(b'\x09\x15\x22\x33\x23\x13\x43\x19\x35\x67\x59\x44\x42\x38\x41'.isascii())
          + Fore.LIGHTGREEN_EX + '>>print(\'\\x80\\x8f\\x94\\x45\\x3a\\xe5\\xb9\\x90\\x28\\xc7\\xd4\\xd6\')\n'
          + Fore.WHITE + '\x80\x8f\x94\x45\x3a\xe5\xb9\x90\x28\xc7\xd4\xd6\n'
          + Fore.LIGHTGREEN_EX + '>>print(b\'\\x80\\x8f\\x94\\x45\\x3a\\xe5\\xb9\\x90\\x28\\xc7\\xd4\\xd6\')\n'
          + Fore.WHITE + '{0}\n'.format(b'\x80\x8f\x94\x45\x3a\xe5\xb9\x90\x28\xc7\xd4\xd6') + Fore.LIGHTGREEN_EX
          + 'print(b\'\\x80\\x8f\\x94\\x45\\x3a\\xe5\\xb9\\x90\\x28\\xc7\\xd4\\xd6\'.isascii())\n' + Fore.WHITE
          + '{0}'.format(b'\x80\x8f\x94\x45\x3a\xe5\xb9\x90\x28\xc7\xd4\xd6'.isascii()))


def describe_bytes_isdigit():
    print(Fore.YELLOW + 'Bytes IsDigit Method Test\n' + Fore.LIGHTGREEN_EX + '>>print(b\'1234\'.isdigit())\n' + Fore.WHITE
          + '{0}\n'.format(b'1234'.isdigit()) + Fore.LIGHTGREEN_EX + '>>print(b\'1.23\'.isdigit())\n' + Fore.WHITE
          + '{0}\n'.format(b'1.23'.isdigit()) + Fore.LIGHTGREEN_EX + '>>a = b\'\u0030\'\n>>b = b\'\u00B2\'\n'
          + '>>print(a.isdigit())')
    a = b'\u0030'
    b = b'\u00B2'
    print('{0}\n'.format(a.isdigit()) + Fore.LIGHTGREEN_EX + '>>print(b.isdigit())\n' + Fore.WHITE
          + '{0}\n'.format(b.isdigit()) + Fore.LIGHTGREEN_EX + '>>a = "\u0030"\n>>b = "\u00B2"\n>>print(a.isdigit())')
    a = '\u0030'
    b = '\u00B2'
    print('{0}\n'.format(a.isdigit()) + Fore.LIGHTGREEN_EX + '>>print(b.isdigit())\n' + Fore.WHITE
          + '{0}'.format(b.isdigit()))


def describe_bytes_islower():
    print(Fore.YELLOW + 'Bytes islower Method Test\n\tFrom https://www.w3schools.com/python/ref_string_islower.asp\n'
          + Fore.LIGHTGREEN_EX + '>>print(b\'hello world\'.islower())\n' + Fore.WHITE
          + '{0}\n'.format(b'hello world'.islower()) + Fore.LIGHTGREEN_EX + '>>print(b\'Hello World\'.islower())\n'
          + Fore.WHITE + '{0}\n'.format(b'Hello world'.islower()) + Fore.LIGHTGREEN_EX + '>>txt = b\'hello world!\'\n'
          + '>>x = txt.islower()\n>>print(x)')
    txt = b'hello world!'
    x = txt.islower()
    print('{0}\n'.format(x) + Fore.LIGHTGREEN_EX + '>>a = b\'Hello world!\'\n>>b = b\'hello 123\'\n'
          + '>>c = b\'mynameisPeter\'')
    a = b'Hello world!'
    b = b'hello 123'
    c = b'mynameisPeter'
    print(Fore.LIGHTGREEN_EX + '>>print(a.islower())\n' + Fore.WHITE + '{0}\n'.format(a.islower()) + Fore.LIGHTGREEN_EX
          + '>>print(b.islower())\n' + Fore.WHITE + '{0}\n'.format(b.islower()) + Fore.LIGHTGREEN_EX
          + '>>print(c.lower())\n' + Fore.WHITE + '{0}'.format(c.islower()))


def describe_bytes_isspace():
    print(Fore.YELLOW + 'Bytes isspace Method Test\n' + Fore.LIGHTGREEN_EX
          + 'print(b\' \\t\\n\\r\\x0b\\f\'.isspace())\n' + Fore.WHITE + '{0}\n'.format(b' \t\n\r\x0b\f'.isspace())
          + Fore.LIGHTGREEN_EX + 'txt = b\'  s  \'\n' + 'x = txt.isspace()\nprint(x)')
    txt = b'  s  '
    x = txt.isspace()
    print(x)


def describe_bytes_istitle():
    print(Fore.YELLOW + 'Bytes istitle Method Test\n' + Fore.LIGHTGREEN_EX + '>>print(b\'Hello World\'.istitle())\n'
          + Fore.WHITE + '{0}\n'.format(b'Hello World'.istitle()) + Fore.LIGHTGREEN_EX
          + '>>print(b\'Hello world\'.istitle())\n' + Fore.WHITE + '{0}\n'.format(b'Hello world'.istitle())
          + Fore.LIGHTGREEN_EX + '>>a = b\'HELLO, AND WELCOME TO MY WORLD\'\n>>b = b\'Hello\'\n>>c = b\'22 Names\'\n'
          + '>>d = b\'This Is %\\\'!?\'\n>>print(a.istitle())')
    a = b'HELLO, AND WELCOME TO MY WORLD'
    b = b'Hello'
    c = b'22 Names'
    d = b'This Is %\'!?'
    print(str(a.istitle()) + Fore.LIGHTGREEN_EX + '\n>>print(b.istitle())\n' + Fore.WHITE + str(b.istitle())
          + Fore.LIGHTGREEN_EX + '\n>>print(c.istitle())\n' + Fore.WHITE + str(c.istitle()) + Fore.LIGHTGREEN_EX
          + '\n>>print(d.istitle())\n' + Fore.WHITE + str(d.istitle()))


def describe_bytes_isupper():
    print(Fore.YELLOW + 'Bytes isupper Method Test\n' + Fore.LIGHTGREEN_EX + '>>print(b\'HELLO WORLD\'.isupper())\n'
          + Fore.WHITE + '{0}\n'.format(b'HELLO WORLD'.isupper()) + Fore.LIGHTGREEN_EX
          + '>>print(b\'Hello world\'.isupper())\n' + Fore.WHITE + '{0}'.format(b'Hello world'.isupper()))


def describe_bytes_join():
    print(Fore.YELLOW + 'Bytes join Method Test\n\tFrom https://www.geeksforgeeks.org/python-string-join-method/\n'
          + Fore.LIGHTGREEN_EX + '>>byte = b\'-\'.join([b\'h\', b\'e\', b\'l\', b\'l\', b\'o\'])\n>>print(str)')
    byte = b'-'.join([b'h', b'e', b'l', b'l', b'o'])
    print('{0}\n'.format(byte) + Fore.LIGHTGREEN_EX + '>>list1 = [b\'g\', b\'e\', b\'e\', b\'k\', b\'s\']\n'
          + '>>print(b"".join(list1))')
    list1 = [b'g', b'e', b'e', b'k', b's']
    print('{0}\n'.format(b''.join(list1)) + Fore.LIGHTGREEN_EX
          + '>>list1 = [b\' \', b\'g\', b\'e\', b\'e\', b\'k\', b\'s\', b\' \']\n>>print("$".join(list1))')
    list1 = [b' ', b'g', b'e', b'e', b'k', b's', b' ']
    print('{0}\n'.format(b'$'.join(list1)) + Fore.LIGHTGREEN_EX + '>>list1 = (b\'1\', b\'2\', b\'3\', b\'4\')\n'
          + '>>s = b\'-\'\n>>s = s.join(list1)>>print(s)')
    list1 = (b'1', b'2', b'3', b'4')
    s = b'-'
    s = s.join(list1)
    print(f'{s}\n' + Fore.LIGHTGREEN_EX + '>>list1 = {b\'1\', b\'2\', b\'3\', b\'4\', b\'4\'}\n>>s = b\'-#-\'\n'
          + '>>s = s.join(list1)\n>>print(s)')
    list1 = {b'1', b'2', b'3', b'4', b'4'}
    s = b'-#-'
    s = s.join(list1)
    print(f'{s}\n' + Fore.LIGHTGREEN_EX + '>>dic = {b\'Geek\': 1, b\'For\': 2, b\'Geeks\': 3}\n'
          + '>>byte = b\'_\'.join(dic)\n>>print(byte)')
    dic = {b'Geek': 1, b'For': 2, b'Geeks': 3}
    byte = b'_'.join(dic)
    print(f'{byte}\n' + Fore.LIGHTGREEN_EX + '>>dic = {1: b\'Geek\', 2: b\'For\', 3: b\'Geeks\'}\n>>import traceback\n'
          + '>>try:\n\tbyte = b\'_\'.join(dic)\nexcept TypeError as e:\n\tprint(traceback.format_exc()')
    dic = {1: b'Geek', 2: b'For', 3: b'Geeks'}
    try:
        byte = b'_'.join(dic)
    except TypeError as e:
        print(traceback.format_exc())
    print(Fore.LIGHTGREEN_EX + '>>words = [b\'apple\', b\'\', b\'banana\', b\'cherry\', b\'\']\n>>separator = b\'@ \'\n'
          + '>>result = separator.join(word for word in words if word)\n>>print(result)')
    words = [b'apple', b'', b'banana', b'cherry', b'']
    separator = b'@ '
    result = separator.join(word for word in words if word)
    print(f'{result}')


def describe_bytes_ljust():
    print(Fore.YELLOW + 'Bytes ljust Method Test\n\tFrom https://www.w3schools.com/python/ref_string_ljust.asp\n'
          + Fore.LIGHTGREEN_EX + '>>txt = b\'banana\'\n>>x = txt.ljust(20)\n>>print(x, b\'is my favorite fruit.\')')
    txt = b'banana'
    x = txt.ljust(20)
    print(x, b'is my favorite fruit.')
    print(Fore.LIGHTGREEN_EX + '>>x = txt.ljust(20, b\'O\')\n>>print(x)')
    x = txt.ljust(20, b'O')
    print(x)


def describe_bytes_lower():
    print(Fore.YELLOW + 'Bytes lower Method Test\n' + Fore.LIGHTGREEN_EX + '>>print(b\'Hello World\'.lower())\n'
          + Fore.WHITE + '{0}'.format(b'Hello World'.lower()))


def describe_bytes_lstrip():
    print(Fore.YELLOW + 'Bytes lstrip Method Test\n' + Fore.LIGHTGREEN_EX + '>>print(b\'   spacious   \'.lstrip())\n'
          + Fore.WHITE + '{0}\n'.format(b'   spacious   '.lstrip()) + Fore.LIGHTGREEN_EX
          + '>>print(b\'www.example.com\'.lstrip(b\'cmowz.\'))\n' + Fore.WHITE
          + '{0}\n'.format(b'www.example.com'.lstrip(b'cmowz.')) + Fore.LIGHTGREEN_EX
          + '>>print(b\'Arthur: three!\'.lstrip(b\'Arthur: \'))\n' + Fore.WHITE
          + '{0}\n'.format(b'Arthur: three!'.lstrip(b'Arthur: ')) + Fore.LIGHTGREEN_EX
          + '>>print(b\'Arthur: three!\'.removeprefix(b\'Arthur: \'))\n' + Fore.WHITE
          + '{0}'.format(b'Arthur: three!'.removeprefix(b'Arthur: ')))


def describe_bytes_partition():
    print(Fore.YELLOW
          + 'Bytes partition Method Test\n\tFrom https://www.geeksforgeeks.org/python-string-partition-method/\n'
          + Fore.LIGHTGREEN_EX + '>>str = b\'I love Geeks for geeks\'\n>>print(str.partition(b\'for\')')
    str = b'I love Geeks for geeks'
    print('{0}\n'.format(str.partition(b'for')) + Fore.LIGHTGREEN_EX + '>>string = b\'light attracts bug\'\n'
          + '# \'attracts\' separator is found\n>>print(string.partition(b\'attracts\'))')
    string = b'light attracts bug'
    # 'attracts' separator is found
    print('{0}\n'.format(string.partition(b'attracts')) + Fore.LIGHTGREEN_EX + '>>string = b\'gold is heavy\'\n'
          + '# \'is\' as partition\n>>print(string.partition(b\'is\'))')
    string = b'gold is heavy'
    print('{0}\n'.format(string.partition(b'is')) + Fore.LIGHTGREEN_EX + '>>string = b\'b follows a, c follows b\'\n'
          + '>>print(string.partition(b\'follow\'))')
    string = b'b follows a, c follows b'
    print('{0}\n'.format(string.partition(b'follows')) + Fore.LIGHTGREEN_EX + '>>string = b\'I am happy, I am proud\'\n'
          + '>>print(string.partition(b\'am\'))')
    string = b'I am happy, I am proud'
    print('{0}\n'.format(string.partition(b'am')) + Fore.LIGHTGREEN_EX + '>>string = b\'geeks for geeks\'\n'
          + '>>print(string.partition(b\'are\'))')
    string = b'geeks for geeks'
    print('{0}\n'.format(string.partition(b'are')) + Fore.LIGHTGREEN_EX
          + '>>url = b\'https://www.example.com/index.html\'\n>>result = url.partition(b\'//\')\n'
          + 'result = result[2].partition(b\'/\')\n>>print(result[0])')
    url = b'https://www.example.com/index.html'
    result = url.partition(b'//')
    result = result[2].partition(b'/')
    print('{0}\n'.format(result[0]) + Fore.LIGHTGREEN_EX
          + '>>sentence = b\'The quick Brown fox jumps over the lazy Fox.\'\n>>result = sentence.partition(b\'fox\')\n'
          + '>>print(result)')
    sentence = b'The quick Brown fox jumps over the lazy Fox.'
    result = sentence.partition(b'fox')
    print('{0}\n'.format(result) + Fore.LIGHTGREEN_EX + '>>result = sentence.partition(b\'Fox\')\n>>print(result)')
    result = sentence.partition(b'Fox')
    print(result)


def describe_bytes_removeprefix():
    print(Fore.YELLOW + 'Bytes removeprefix Method Test\n' + Fore.LIGHTGREEN_EX
          + '>>print(b\'TestHook\'.removeprefix(b\'Test\'))\n' + Fore.WHITE
          + '{0}\n'.format(b'TestHook'.removeprefix(b'Test')) + Fore.LIGHTGREEN_EX
          + '>>print(b\'BaseTestCase\'.removeprefix(b\'Test\'))\n' + Fore.WHITE
          + '{0}'.format(b'BaseTestCase'.removeprefix(b'Test')))


def describe_bytes_removesuffix():
    print(Fore.YELLOW + 'Bytes removesuffix Method Test\n' + Fore.LIGHTGREEN_EX
          + '>>print(b\'MiscTests\'.removesuffix(b\'Tests\'))\n' + Fore.WHITE
          + '{0}\n'.format(b'MiscTests'.removesuffix(b'Tests')) + Fore.LIGHTGREEN_EX
          + '>>print(b\'TmpDirMixin\'.removesuffix(b\'Tests\')\n' + Fore.WHITE
          + '{0}'.format(b'TmpDirMixin'.removesuffix(b'Tests')))


def describe_bytes_replace():
    print(Fore.YELLOW + 'Bytes replace Method Test\n\tFrom https://www.geeksforgeeks.org/python-string-replace/'
          + Fore.LIGHTGREEN_EX + '>>string = b\'Hello World\'\n>>new_string = string.replace(b\'Hello\', b\'Good Bye\')'
          + '\n>>print(new_string)')
    string = b'Hello world'
    new_string = string.replace(b'Hello', b'Good Bye')
    print(f'{new_string}\n' + Fore.LIGHTGREEN_EX + '>>string = b\'Replace\'\n'
          + '>>new_string = string.replace(b\'Replace\', b\'Replaced\')\n>>print(new_string)')
    string = b'Replace'
    new_string = string.replace(b'Replace', b'Replaced')
    print(f'{new_string}\n' + Fore.LIGHTGREEN_EX + '>>string = b\'Good Morning\'\n'
          + '>>new_string = string.replace(b\'Good\', b\'Great\')\n>>print(new_string)')
    string = b'Good Morning'
    new_string = string.replace(b'Good', b'Great')
    print(f'{new_string}\n' + Fore.LIGHTGREEN_EX + '>>string = b\'grrks FOR grrks\'\n'
          + '# replace all instances of b\'r\' (old) with b\'e\' (new)\n>>new_string = string.replace(b\'r\', b\'e\')')
    string = b'grrks FOR grrks'
    # replace all instances of b'r' (old) with b'e' (new)
    new_string = string.replace(b'r', b'e')
    print(Fore.LIGHTGREEN_EX + '>>print(string)\n' + Fore.WHITE + f'{string}\n' + Fore.LIGHTGREEN_EX
          + '>>print(new_string)\n' + Fore.WHITE + f'{new_string}\n' + Fore.LIGHTGREEN_EX
          + '>>string = b\'geeks for geeks \\ngeeks for geeks\'\n>>print(string)')
    string = b'geeks for geeks \ngeeks for geeks'
    print(f'{string}\n' + Fore.LIGHTGREEN_EX + '# Prints the string by replacing only\n# 3 occurrence of Geeks\n'
          + '>>print(string.replace(b\'geeks\', b\'GeeksforGeeks\'))\n' + Fore.WHITE
          + '{0}\n'.format(string.replace(b'geeks', b'GeeksforGeeks')) + Fore.LIGHTGREEN_EX
          + '>>string = b\'geeks for geeks geeks geeks geeks\'\n# Prints the string by replacing b\'e\' by b\'a\'\n'
          + '>>print(string.replace(b\'e\', b\'a\')')
    string = b'geeks for geeks geeks geeks geeks'
    print('{0}\n'.format(string.replace(b'e', b'a')) + Fore.LIGHTGREEN_EX
          + '# Prints the string by replacing only 3 occurrence of b\'ek\' by b\'a\'\n'
          + '>>print(string.replace(b\'ek\', b\'a\', 3))\n' + Fore.WHITE
          + '{0}\n'.format(string.replace(b'ek', b'a', 3))
          + Fore.LIGHTGREEN_EX + '>>my_string = b\'geeks for geeks \'\n>>old_substring = b\'k\'\n'
          + '>>new_substring = b\'x\'\n>>split_list = my_string.split(old_substring)\n'
          + '>>new_list = [new_substring if i < len(split_list)-1 else b\'\' for i in range(len(split_list)-1)]\n'
          + '>>new_string = b\'\'.join([split_list[i] + new_list[i] for i in range(len(split_list)-1)]'
          + ' + [split_list[-1]])\n>>print(new_string)')
    my_string = b'geeks for geeks'
    old_substring = b'k'
    new_substring = b'x'
    split_list = my_string.split(old_substring)
    new_list = [new_substring if i < len(split_list)-1 else b'' for i in range(len(split_list)-1)]
    new_string = b''.join([split_list[i] + new_list[i] for i in range(len(split_list)-1)] + [split_list[-1]])
    print(new_string)


def describe_bytes_rfind():
    print(Fore.YELLOW + 'Bytes rfind Method Test\n\tFrom https://www.w3schools.com/python/ref_string_rfind.asp\n'
          + Fore.LIGHTGREEN_EX + '>>txt = b\'Mi casa, su casa\'\n>>x = txt.rfind(b\'casa\')\n>>print(x)')
    txt = b'Mi casa, su casa'
    x = txt.rfind(b'casa')
    print(f'{x}\n' + Fore.LIGHTGREEN_EX + '>>txt = b\'Hello, welcome to my world.\'\n>>x = txt.rfind(b\'e\')\n'
          + '>>print(x)')
    txt = b'Hello, welcome to my world.'
    x = txt.rfind(b'e')
    print(f'{x}\n' + Fore.LIGHTGREEN_EX + '>>x = txt.rfind(b\'e\', 5, 10)\n>>print(x)')
    x = txt.rfind(b'e', 5, 10)
    print(f'{x}\n' + Fore.LIGHTGREEN_EX + '>>print(txt.rfind(b\'q\')\n' + Fore.WHITE + '{0}\n'.format(txt.rfind(b'q'))
          + Fore.LIGHTGREEN_EX
          + '>>try:\n\tprint(txt.rindex(b\'q\')\n  except ValueError as e:\n\tprint(traceback.format_exc()')
    try:
        print(txt.rindex(b'q'))
    except ValueError as e:
        print(traceback.format_exc())


def describe_bytes_rindex():
    print(Fore.YELLOW + 'Bytes rindex Method Test\n\tFrom https://www.geeksforgeeks.org/python-string-rindex-method/'
          + Fore.LIGHTGREEN_EX + '>>text = b\'geeks for geeks\'\n>>result = text.rindex(b\'geeks\')\n'
          + '>>print("Sub byte b\'geeks\':", result)')
    text = b'geeks for geeks'
    result = text.rindex(b'geeks')
    print(f'Sub byte b\'geeks\': {result}\n' + Fore.LIGHTGREEN_EX + '>>string = b\'ring ring\'\n'
          + '# checks for the substring in the range 0-4 of the string\n>>print(string.rindex(b\'ring\', 0, 4))\n')
    string = b'ring ring'
    # checks for the substring in the range 0-4of the string
    print('{0}\n'.format(string.rindex(b'ring', 0, 4)) + Fore.LIGHTGREEN_EX
          + '# same as using 0 & 4 as start, end value\n>>print(string.rindex(b\'ring\', 0 -5))')
    # same as using 0 & 4 as start, end value
    print('{0}\n'.format(string.rindex(b'ring', 0, -5)) + Fore.LIGHTGREEN_EX + '>>string = b\'101001010\'\n'
          + '# since there are no b\'101\' substring after string[0:3]\n'
          + '# thus it will take the last occurrence of b\'101\'\n>>print(string.rindex(\'101\', 2))')
    string = b'101001010'
    # since there are no b'101' substring after string[0:3]
    # thus it will take the last occurrence of b'101'
    print('{0}\n'.format(string.rindex(b'101', 2)) + Fore.LIGHTGREEN_EX + '>>string = b\'ring ring\'\n'
          + '# search for the substring,\n# from right in the whole string\n>>print(string.rindex(b\'ring\')')
    string = b'ring ring'
    # search for the substring,
    # from right in the whole string
    print('{0}\n'.format(string.rindex(b'ring')) + Fore.LIGHTGREEN_EX + '>>string = b\'geeks\'\n'
          + '# this will return the right-most b\'e\'\n>>print(string.rindex(b\'e\'))')
    string = b'geeks'
    # this will return the right-most b'e'
    print('{0}\n'.format(string.rindex(b'e')) + Fore.LIGHTGREEN_EX + '# Python code to demonstrate error by rindex()\n'
          + '>>text = b\'geeks for geeks\'\n>>try:\n\tresult = text.rindex(b\'pawan\')\n  except ValueError as e:\n\t'
          + 'print(traceback.format_exc()\n>>print("Substring b\'pawan\':", result)')
    # Python code to demonstrate error by rindex()
    text = b'geeks for geeks'
    try:
        result = text.rindex(b'pawan')
    except ValueError as e:
        print(traceback.format_exc())


def describe_bytes_rjust():
    print(Fore.YELLOW + 'Bytes rjust Method Test\n\tFrom '
          + 'https://www.programiz.com/python-programming/methods/string/rjust\n' + Fore.LIGHTGREEN_EX
          + '>>text = b\'Python\'\n# right aligns b\'Python\' up to width 10 using b\'*\'\n'
          + '>>result = text.rjust(10, b\'*\')\n>>print(result)')
    text = b'Python'
    # right aligns b'Python' up to width 10 using b'*'
    result = text.rjust(10, b'*')
    print(f'{result}\n' + Fore.LIGHTGREEN_EX + '>>text = b\'programming\'\n'
          + '# right aligns text up to length 15 using b\'$\'\n>>result = text.rjust(15, b\'$\')\n>>print(result)')
    text = b'programming'
    # right aligns text up to length 15 using b'$'
    result = text.rjust(15, b'$')
    print(f'{result}\n' + Fore.LIGHTGREEN_EX + '>>text = b\'cat\'\n# passing only width and not specifying fillchar\n'
          + '>>result = text.rjust(7)\n>>print(result)')
    text = b'cat'
    # passing only width and not specifying fillchar
    result = text.rjust(7)
    print(f'{result}\n' + Fore.LIGHTGREEN_EX + '>>text = b\'Ninja Turtles\'\n# width equal to length of string\n'
          + '>>result1 = text.rjust(13, b\'*\')\n>>print(result1)')
    text = b'Ninja Turtles'
    # width equal to length of string
    result1 = text.rjust(13, b'*')
    print(f'{result1}\n' + Fore.LIGHTGREEN_EX + '# width less than length of string\n>>result2 = text.rjust(10, b\'*\')'
          + '\n>>print(result2)')
    result2 = text.rjust(10, b'*')
    print(f'{result2}')


def describe_bytes_rpartition():
    print(Fore.YELLOW + 'Bytes rpartition Method Test\n\tFrom '
          + 'https://www.geeksforgeeks.org/python-string-rpartition-method/\n' + Fore.LIGHTGREEN_EX
          + '# String need to split\n>>string1 = b\'Geeks@for@Geeks@is@for@geeks\'\n'
          + '>>string2 = b\'Ram is not eating but Mohan is eating\'\n# Here b\'@\' is a separator\n'
          + '>>print(string1.rpartition(b\'@\'))')
    # String need to split
    string1 = b'Geeks@for@Geeks@is@for@geeks'
    string2 = b'Ram is not eating but Mohan is eating'
    # Here b'@' is a separator
    print('{0}\n'.format(string1.rpartition(b'@')) + Fore.LIGHTGREEN_EX + '# Here b\'is\' is separator\n'
          + '>>print(string2.rpartition(b\'is\'))')
    # Here b'is' is separator
    print('{0}\n'.format(string2.rpartition(b'is')) + Fore.LIGHTGREEN_EX + '# String need to split\n'
          + '>>string = b\'Sita is going to school\'\n# Here b\'not\' is a separator which is not\n'
          + '# present in the given string\n>>print(string.rpartition(b\'not\'))')
    # String need to split
    string = b'Sita is going to school'
    # Here b'not' is a separator which is not
    # present in the given string
    print('{0}\n'.format(string.rpartition(b'not')) + Fore.LIGHTGREEN_EX + '# Python3 code explaining TypeError\n'
          + '# in rpartition()\n>>string = b\'Bruce Waine is Batman\'\n# Nothing is passed as separator\n'
          + '>>try:\n\tprint(string.rpartition())\n  except TypeError as e:\n\tprint(traceback.format_exc())')
    # Python3 code explaining TypeError
    # in rpartition()
    string = b'Bruce Waine is Batman'
    # Nothing is passed as separator
    try:
        print(string.rpartition())
    except TypeError as e:
        print(traceback.format_exc())
    print(Fore.LIGHTGREEN_EX + '>>try:\n\tprint(string.rpartition(b''))\n  except ValueError as e:\n\t'
          + 'print(traceback.format_exc())')
    try:
        print(string.rpartition(b''))
    except ValueError as e:
        print(traceback.format_exc())


def describe_bytes_rsplit():
    print(Fore.YELLOW + 'Bytes rsplit Method Test\n\t'
          + 'From https://www.programiz.com/python-programming/methods/string/rsplit\n' + Fore.LIGHTGREEN_EX
          + '>>text = b\'Love thy neighbor\'\n# splits at space\n>>text.rsplit()')
    text = b'Love thy neighbor'
    # splits at space
    print(f'{text.rsplit()}\n' + Fore.LIGHTGREEN_EX + '>>grocery = b\'Milk, Chicken, Bread\'\n# splits at b\',\'\n'
          + '>>grocery.rsplit(b\', \')')
    grocery = b'Milk, Chicken, Bread'
    # splits at b','
    print('{0}\n'.format(grocery.rsplit(b', ')) + Fore.LIGHTGREEN_EX + '# Splitting at b\':\'\n'
          + '>>grocery.rsplit(b\':\')')
    # Splitting at b':'
    print('{0}\n'.format(grocery.rsplit(b':')) + Fore.LIGHTGREEN_EX + '>>grocery = b\'Milk, Chicken, Bread, Butter\'\n'
          + '# maxsplit: 2\n>>grocery.rsplit(b\', \', 2)')
    grocery = b'Milk, Chicken, Bread, Butter'
    # maxsplit: 2
    print('{0}\n'.format(grocery.rsplit(b', ', 2)) + Fore.LIGHTGREEN_EX + '# maxsplit: 1\n>>grocery.rsplit(b\', \', 1)')
    # maxsplit: 1
    print('{0}\n'.format(grocery.rsplit(b', ', 1)) + Fore.LIGHTGREEN_EX + '# maxsplit: 5\n>>grocery.rsplit(b\', \', 5)')
    # maxsplit: 5
    print('{0}\n'.format(grocery.rsplit(b', ', 5)) + Fore.LIGHTGREEN_EX + '# maxsplit: 0\n>>grocery.rsplit(b\', \', 0)')
    # maxsplit: 0
    print('{0}'.format(grocery.rsplit(b', ', 0)))


def describe_bytes_rstrip():
    print(Fore.YELLOW + 'Bytes rstrip Method Test\n' + Fore.LIGHTGREEN_EX + '>>b\'   spacious  \'.rstrip()\n'
          + Fore.WHITE + '{0}\n'.format(b'   spacious   '.rstrip()) + Fore.LIGHTGREEN_EX
          + '>>b\'mississippi\'.rstrip(b\'ipz\')\n' + Fore.WHITE + '{0}\n'.format(b'mississippi'.rstrip(b'ipz'))
          + Fore.LIGHTGREEN_EX + '>>b\'Monty Python\'.rstrip(b\' Python\')\n' + Fore.WHITE
          + '{0}\n'.format(b'Monty Python'.rstrip(b' Python')) + Fore.LIGHTGREEN_EX
          + 'b\'Monty Python\'.removesuffix(b\' Python\')\n' + Fore.WHITE
          + '{0}'.format(b'Monty Python'.removesuffix(b' Python')))


def describe_bytes_split():
    print(Fore.YELLOW + 'Bytes split Method Test\n' + Fore.LIGHTGREEN_EX + '>>b\'1,,2\'.split(b\',\')\n' + Fore.WHITE
          + '{0}\n'.format(b'1,,2'.split(b',')) + Fore.LIGHTGREEN_EX + '>>b\'1<>2<>3\'.split(b\'<>\')\n' + Fore.WHITE
          + '{0}\n'.format(b'1<>2<>3'.split(b'<>')) + Fore.LIGHTGREEN_EX + '>>b\'1,2,3\'.split(b\',\')\n' + Fore.WHITE
          + '{0}\n'.format(b'1,2,3'.split(b',')) + Fore.LIGHTGREEN_EX + '>>b\'1,2,3\'.split(b\',\', maxsplit=1)\n'
          + Fore.WHITE + '{0}\n'.format(b'1,2,3'.split(b',', maxsplit=1)) + Fore.LIGHTGREEN_EX
          + '>>b\'1,2,,3,\'.split(b\',\')\n' + Fore.WHITE + '{0}\n'.format(b'1,2,,3,'.split(b',')) + Fore.LIGHTGREEN_EX
          + '>>b\'1 2 3\'.split()\n' + Fore.WHITE + '{0}\n'.format(b'1 2 3'.split()) + Fore.LIGHTGREEN_EX
          + '>>b\'1 2 3\'.split(maxsplit=1)\n' + Fore.WHITE + '{0}\n'.format(b'1 2 3'.split(maxsplit=1))
          + Fore.LIGHTGREEN_EX + '>>b\'   1   2   3   \'.split()\n' + Fore.WHITE
          + '{0}'.format(b'   1   2   3   '.split()))


def describe_bytes_splitlines():
    print(Fore.YELLOW + 'Bytes splitlines Method Test\n' + Fore.LIGHTGREEN_EX
          + '>>b\'ab c\\n\\nde fg\\rkl\\r\\n\'.splitlines()\n' + Fore.WHITE
          + '{0}\n'.format(b'ab c\n\nde fg\rkl\r\n'.splitlines()) + Fore.LIGHTGREEN_EX
          + '>>b\'ab c\\n\\nde fg\\rkl\\r\\n\'.splitlines(keepends=True)\n' + Fore.WHITE
          + '{0}\n'.format(b'ab c\n\nde fg\rkl\r\n'.splitlines(keepends=True)) + Fore.LIGHTGREEN_EX
          + '>>b"".split(b\'\\n\'), b"Two lines\\n".split(b\'\\n\')\n' + Fore.WHITE
          + '{0}\n'.format((b''.split(b'\n'), b'Two lines\n'.split(b'\n'))) + Fore.LIGHTGREEN_EX
          + '>>b\'\'.splitlines(), b\'One line\\n\'.splitlines()\n' + Fore.WHITE
          + '{0}'.format((b''.splitlines(), b'One line\n'.splitlines())))


def describe_bytes_startswith():
    print(Fore.YELLOW + 'Bytes startswith Method Test\n\t'
          + 'From https://www.programiz.com/python-programming/methods/string/startswith\n' + Fore.LIGHTGREEN_EX
          + '>>message = b\'Python is fun\'\n>>message.startswith(b\'Python\')')
    message = b'Python is fun'
    print('{0}\n'.format(message.startswith(b'Python')) + Fore.LIGHTGREEN_EX
          + '>>text = b\'Python is easy to learn.\'\n>>result = text.startswith(b\'is easy\')\n>>print(result)')
    text = b'Python is easy to learn.'
    result = text.startswith(b'is easy')
    print(f'{result}\n' + Fore.LIGHTGREEN_EX + '>>result = text.startswith(b\'Python is \')\n>>print(result)')
    result = text.startswith(b'Python is ')
    print(f'{result}\n' + Fore.LIGHTGREEN_EX + '>>result = text.startswith(b\'Python is easy to learn.\')\n'
          + '>>print(result)')
    result = text.startswith(b'Python is easy to learn.')
    print(f'{result}\n' + Fore.LIGHTGREEN_EX + '>>text = b\'Python programming is easy.\'\n# start parameter: 7\n'
          + '# b\'programming is easy.\' string is searched\n>>result = text.startswith(b\'programming is\', 7)\n'
          + '>>print(result)')
    text = b'Python programming is easy.'
    # start parameter: 7
    # b'programming is easy.' string is searched
    result = text.startswith(b'programming is', 7)
    print(f'{result}\n' + Fore.LIGHTGREEN_EX + '# start: 7, end: 18\n# b\'programming\' string is searched\n'
          + '>>result = text.startswith(b\'programming is\', 7, 18)\n>>print(result)')
    # start: 7, end: 18
    # b'programming' string is searched
    result = text.startswith(b'programming is', 7, 18)
    print(f'{result}\n' + Fore.LIGHTGREEN_EX + '>>result = text.startswith(b\'program\', 7, 18)\n>>print(result)')
    result = text.startswith(b'program', 7, 18)
    print(f'{result}\n' + Fore.LIGHTGREEN_EX + '>>text = b\'programming is easy\'\n'
          + '>>result = text.startswith((b\'python\', b\'programming\'))\n>>print(result)')
    text = b'programming is easy'
    result = text.startswith((b'python', b'programming'))
    print(f'{result}\n' + Fore.LIGHTGREEN_EX + '>>result = text.startswith((b\'is\', b\'easy\', b\'java\'))\n'
          + '>>print(result)')
    result = text.startswith((b'is', b'easy', b'java'))
    print(f'{result}\n' + Fore.LIGHTGREEN_EX + '# With start and end parameter\n# b\'is easy\' string is checked\n'
          + '>>result = text.startswith((b\'programming\', b\'easy\'), 12, 19)\n>>print(result)')
    result = text.startswith((b'programming', b'easy'), 12, 19)
    print(f'{result}')


def describe_bytes_strip():
    print(Fore.YELLOW + 'Bytes strip Method Test\n' + Fore.LIGHTGREEN_EX + '>>b\'   spacious   \'.strip()\n'
          + Fore.WHITE + '{0}\n'.format(b'   spacious   '.strip()) + Fore.LIGHTGREEN_EX
          + '>>b\'www.example.com\'.strip(b\'cmowz.\')\n' + Fore.WHITE
          + '{0}'.format(b'www.example.com'.strip(b'cmowz.')))


def describe_bytes_swapcase():
    print(Fore.YELLOW + 'Bytes swapcase Method Test\n' + Fore.LIGHTGREEN_EX + '>>b\'Hello World\'.swapcase()\n'
          + Fore.WHITE + '{0}'.format(b'Hello World'.swapcase()))


def titlecase(s):
    return re.sub(rb"[A-Za-z]+('[A-Za-z]+)?", lambda mo: mo.group(0)[0:1].upper() + mo.group(0)[1:].lower(), s)


def describe_bytes_title():
    print(Fore.YELLOW + 'Bytes title Method Test\n' + Fore.LIGHTGREEN_EX + '>>b\'Hello world\'.title()\n' + Fore.WHITE
          + '{0}\n'.format(b'Hello world'.title()) + Fore.LIGHTGREEN_EX
          + '>>b\'they\'re bill\'s friends from the UK\'.title()\n' + Fore.WHITE
          + '{0}\n'.format(b"they're bill's friends from the UK".title()) + Fore.LIGHTGREEN_EX + '>>import re\n'
          + '>>def titlecase(s):\n\treturn re.sub(rb"[A-Za-z]+(\'[A-Za-z]+)?",'
          + ' lambda mo: mo.group(0)[0:1].upper() + mo.group(0)[1:].lower(), s)\n'
          + '>>titlecase(b"they\'re bill\'s friends.")\n' + Fore.WHITE
          + '{0}'.format(titlecase(b"they're bill's friends.")))


def describe_bytes_translate():
    print(Fore.YELLOW + 'Bytes translate Method Test\n' + Fore.LIGHTGREEN_EX
          + '>>b\'read this short text\'.translate(None, b\'aeiou\')\n' + Fore.WHITE
          + '{0}'.format(b'read this short text'.translate(None, b'aeiou')))


def describe_bytes_upper():
    print(Fore.YELLOW + 'Bytes upper Method Test\n' + Fore.LIGHTGREEN_EX + '>>b\'Hello World\'.upper()\n' + Fore.WHITE
          + '{0}'.format(b'Hello World'.upper()))


def describe_bytes_zfill():
    print(Fore.YELLOW + 'Bytes zfill Method Test\n' + Fore.LIGHTGREEN_EX + '>>b\'42\'.zfill(5)\n' + Fore.WHITE
          + '{0}\n'.format(b'42'.zfill(5)) + Fore.LIGHTGREEN_EX + '>>b\'-42\'.zfill(5)\n' + Fore.WHITE
          + '{0}'.format(b'-42'.zfill(5)))


def describe_conjugate():
    print(Fore.YELLOW + 'conjugate Method Test\n\tFrom https://www.geeksforgeeks.org/numpy-conj-in-python/\n'
          + Fore.LIGHTGREEN_EX + '>>in_complx1 = 2+4j\n>>out_complx1 = in_complx1.conjugate()\n'
          + '>>print("Output conjugated complex number of 2+4j : ", out_complx1)')
    in_complx1 = 2 + 4j
    out_complx1 = in_complx1.conjugate()
    print(f'Output conjugated complex number of {in_complx1} : {out_complx1}\n' + Fore.LIGHTGREEN_EX
          + '>>in_complx2 = 5 - 8j\n>>out_complx2 = in_complx2.conjugate()\n'
          + '>>print("Output conjugated complex number of 5-8j: ", out_complx2)')
    in_complx2 = 5 - 8j
    out_complx2 = in_complx2.conjugate()
    print(f'Output conjugated complex number of {in_complx2} : {out_complx2}\n' + Fore.LIGHTGREEN_EX
          + '>>import numpy as np\n>>in_array = np.eye(2) + 3j * np.eye(2)\n>>print("Input array: \\n", in_array)')
    in_array = np.eye(2) + 3j * np.eye(2)
    print(f'Input array :\n{in_array}\n' + Fore.LIGHTGREEN_EX + '>>out_array = in_array.conjugate()\n'
          + '>>print("Output conjugated array : \\n", out_array)')
    out_array = in_array.conjugate()
    print(f'Output conjugated array :\n{out_array}')


def describe_memoryview_c_contiguous():
    print(Fore.YELLOW + 'MemoryView c_contiguous Method Test\n' + Fore.LIGHTGREEN_EX
          + '>>in_array = np.eye(3) + 5j * np.eye(3)\n>>m = memoryview(in_array)\n>>info = m.c_contiguous\n'
          + '>>print(info)')
    in_array = np.eye(3) + 5j * np.eye(3)
    m = memoryview(in_array)
    info = m.c_contiguous
    print(info)


def geeks_function(x):
    return x


# Python program to illustrate
# callable() a test function
def geek_return_five(x):
    return 5


def describe_callable():
    print(Fore.YELLOW + 'callable Method Test\nFrom https://www.geeksforgeeks.org/callable-in-python/\n'
          + Fore.LIGHTGREEN_EX + '>>x = 10\n>>print(callable(x))')
    x = 10
    print(f'{callable(x)}\n' + Fore.LIGHTGREEN_EX + '>>def geeks_function(x):\n\treturn x\n>>y = geeks_function\n'
          + '>>print(callable(y))')
    y = geeks_function
    print(f'{callable(y)}\n' + Fore.LIGHTGREEN_EX + '# Python program to illustrate\n# callable() a test function\n'
          + '>>def geek_return_five():\n\treturn 5\n# an object is created of Geek()\n>>let = geek_return_five\n'
          + '>>print(callable(let))')
    # an object is created of Geek()
    let = geek_return_five
    print(f'{callable(let)}\n' + Fore.LIGHTGREEN_EX + '# a test variable\n>>num = 5 * 5\n>>print(callable(num))')
    num = 5 * 5
    print(f'{callable(num)}\n' + Fore.LIGHTGREEN_EX + '# Python program to illustrate callable()\n>>class GeekObj:\n\t'
          + 'def __call__(self):\n\t\tprint(\'Hello GeeksforGeeks\')\n'
          + '# Suggests that the GeekObj class is callable\n>>print(callable(GeekObj))\n' + Fore.WHITE
          + f'{callable(GeekObj)}\n' + Fore.LIGHTGREEN_EX + '# This proves that class is callable\n'
          + '>>geek_object = GeekObj()\n>>geek_object()')
    # Suggests that the Geek class is callable
    # This proves that class is callable
    geek_object = GeekObj
    geek_object()
    print(Fore.LIGHTGREEN_EX + '# Python program to illustrate callable()\n>>class GeekObjTwo:\n\tdef testFunc(self):'
          + '\n\t\tprint(\'Hello GeeksforGeeks\')\n# Suggests that the GeekObjTwo class is callable\n'
          + '>>print(callable(GeekObjTwo))\n' + Fore.WHITE + f'{callable(GeekObjTwo)}\n' + Fore.LIGHTGREEN_EX
          + '>>GeekObject = GeekObjTwo()\n# The object will be created but\n# returns an error on calling\n'
          + '>>try:\n\tGeekObject()\n  except TypeError as e:\n\tprint(traceback.format_exc())')
    # Suggests that the GeekObjTwo class is callable
    GeekObject = GeekObjTwo()
    # The object will be created but
    # returns an error on calling
    try:
        GeekObject()
    except TypeError as e:
        print(traceback.format_exc())


def describe_cast():
    print(Fore.YELLOW + 'Cast Method Test\n' + Fore.LIGHTGREEN_EX + '>>import array\n'
          + '>>a = array.array(\'l\', [1, 2, 3])\n>>x = memoryview(a)\n>>x.format')
    a = array.array('l', [1, 2, 3])
    x = memoryview(a)
    print(f'{x.format}\n' + Fore.LIGHTGREEN_EX + '>>x.itemsize\n' + Fore.WHITE + f'{x.itemsize}\n' + Fore.LIGHTGREEN_EX
          + '>>len(x)\n' + Fore.WHITE + f'{len(x)}\n' + Fore.LIGHTGREEN_EX + '>>x.nbytes\n' + Fore.WHITE
          + f'{x.nbytes}\n' + Fore.LIGHTGREEN_EX + '>>y = x.cast(\'B\')\n>>y.format')
    y = x.cast('B')
    print(f'{y.format}\n' + Fore.LIGHTGREEN_EX + '>>y.itemsize\n' + Fore.WHITE + f'{y.itemsize}\n' + Fore.LIGHTGREEN_EX
          + '>>len(y)\n' + Fore.WHITE + f'{len(y)}\n' + Fore.LIGHTGREEN_EX + '>>y.nbytes\n' + Fore.WHITE
          + f'{y.nbytes}\n' + Fore.LIGHTGREEN_EX + '>>b = bytearray(b\'zyz\')\n>>x = memoryviw(b)\n>>try:\n\t'
          + 'x[0] = b\'a\'\nexcept TypeError as e:\n\tprint(traceback.format_exc())')
    b = bytearray(b'zyz')
    x = memoryview(b)
    try:
        x[0] = b'a'
    except TypeError as e:
        print(traceback.format_exc())
    print(Fore.LIGHTGREEN_EX + '>>y = x.cast(\'c\')\n>>y[0] = b\'a\'\n>>b')
    y = x.cast('c')
    y[0] = b'a'
    print(f'{b}\n' + Fore.LIGHTGREEN_EX + '>>import struct\n>>buf = struct.pack("i"*12, *list(range(12)))\n'
          + '>>x = memoryview(buf)\n>>y = x.cast(\'i\', shape=[2, 2, 3])\n>>y.tolist()')
    buf = struct.pack("i"*12, *list(range(12)))
    x = memoryview(buf)
    y = x.cast('i', shape=[2, 2, 3])
    print(f'{y.tolist()}\n' + Fore.LIGHTGREEN_EX + '>>y.format\n' + Fore.WHITE + f'{y.format}\n' + Fore.LIGHTGREEN_EX
          + '>>y.itemsize\n' + Fore.WHITE + f'{y.itemsize}\n' + Fore.LIGHTGREEN_EX + '>>len(y)\n' + Fore.WHITE
          + f'{len(y)}\n' + Fore.LIGHTGREEN_EX + '>>y.nbytes\n' + Fore.WHITE + f'{y.nbytes}\n' + Fore.LIGHTGREEN_EX
          + '>>z = y.cast(\'b\')\n>>z.format')
    z = y.cast('b')
    print(f'{z.format}\n' + Fore.LIGHTGREEN_EX + '>>z.itemsize\n' + Fore.WHITE + f'{z.itemsize}\n' + Fore.LIGHTGREEN_EX
          + '>>len(z)\n' + Fore.WHITE + f'{len(z)}\n' + Fore.LIGHTGREEN_EX + '>>z.nbytes\n' + Fore.WHITE
          + f'{z.nbytes}\n' + Fore.LIGHTGREEN_EX + '>>buf = struct.pack("L"*6, *list(range(6)))\n'
          + '>>x = memoryview(buf)\n>>y = x.cast(\'L\', shape=[2, 3])\n>>len(y)')
    buf = struct.pack("L"*6, *list(range(6)))
    x = memoryview(buf)
    y = x.cast('L', shape=[2, 3])
    print(f'{len(y)}\n' + Fore.LIGHTGREEN_EX + '>>y.nbytes\n' + Fore.WHITE + f'{y.nbytes}\n' + Fore.LIGHTGREEN_EX
          + '>>y.tolist()\n' + Fore.WHITE + f'{y.tolist()}')


def describe_chr():
    print(Fore.YELLOW + 'chr Method Test\n' + Fore.LIGHTGREEN_EX + '>>chr(97)\n' + Fore.WHITE + f'{chr(97)}\n'
          + Fore.LIGHTGREEN_EX + '>>chr(8364)\n' + Fore.WHITE + f'{chr(8364)}\n' + Fore.LIGHTGREEN_EX + '>>ord(\'€\')\n'
          + Fore.WHITE + '{0}'.format(ord('€')))


def describe_classmethod_float_fromhex():
    print(Fore.YELLOW + 'classmethod float fromhex Method Test\n' + Fore.LIGHTGREEN_EX
          + '>>number = (3 + 10./16 + 7./16**2) * 2.0**10')
    number = (3 + 10./16 + 7./16**2) * 2**10
    print(f'{number}\n' + Fore.LIGHTGREEN_EX + '>>float.fromhex(\'0x3.a7p10\')\n' + Fore.WHITE
          + '{0}\n'.format(float.fromhex('0x3.a7p10')) + Fore.LIGHTGREEN_EX + '>>float.hex(3740.0)\n' + Fore.WHITE
          + f'{float.hex(3740.0)}')


def describe_classmethod_bytes_fromhex_string():
    print(Fore.YELLOW + 'classmethod bytes fromhex string Method Test\n' + Fore.LIGHTGREEN_EX
          + '>>bytes.fromhex(\'2Ef0 F1F2  \')\n' + Fore.WHITE + '{0}\n'.format(bytes.fromhex('2Ef0 F1F2  ')))


def describe_classmethod_fromkeys():
    print(Fore.YELLOW + 'classmethod fromkeys Method Test\n'
          + 'From https://www.geeksforgeeks.org/python-dictionary-fromkeys-method/' + Fore.LIGHTGREEN_EX
          + '>>seq = (\'a\', \'b\', \'c\')\n>>print(dict.fromkeys(seq, None))')
    seq = ('a', 'b', 'c')
    print(f'{dict.fromkeys(seq, None)}\n' + Fore.LIGHTGREEN_EX + '>>seq = {\'a\', \'b\', \'c\', \'d\', \'e\'}\n'
          + '# creating dict with default values as None\n>>res_dict = dict.fromkeys(seq)\n'
          + '>>print("The newly created dict with None values :" + str(res_dict))\n')
    seq = {'a', 'b', 'c', 'd', 'e'}
    # creating dict with default values as None
    res_dict = dict.fromkeys(seq)
    print("The newly created dict with None value :" + str(res_dict) + Fore.LIGHTGREEN_EX
          + '\n# creating dict with default values as 1\n>>res_dict2 = dict.fromkeys(seq, 1)\n'
          + '>>print("The newly created dict with 1 as value : " + str(res_dict2))\n')
    # creating dict with default values as 1
    res_dict2 = dict.fromkeys(seq, 1)
    print("The newly created dict with 1 as value : " + str(res_dict2) + Fore.LIGHTGREEN_EX
          + '\n>>lis1 = [2, 3]\n>>res_dict = dict.fromkeys(seq, lis1)\n# Printing created dict\n'
          + '>>print("The newly created dict with list values : " + str(res_dict))')
    lis1 = [2, 3]
    res_dict = dict.fromkeys(seq, lis1)
    # Printing created dict
    print("The newly created dict with list values : " + str(res_dict) + Fore.LIGHTGREEN_EX + '\n# appending to lis1\n'
          + '>>lis1.append(4)\n>>print("The dict with list values after appending : ", str(res_dict))')
    # appending to lis1
    lis1.append(4)
    print("The dict with list values after appending : " + str(res_dict) + Fore.LIGHTGREEN_EX + '\n>>lis1 = [2, 3]\n'
          + '# using fromkeys() to convert sequence to dict\n# using dict. comprehension\n'
          + '>>res_dict2 = {key: list(lis1) for key in seq}\n# Printing created dict\n'
          + '>>print("The newly created dict with list values : " + str(res_dict2))')
    lis1 = [2, 3]
    # using fromkeys() to convert sequence to dict
    # using dict. comprehension
    res_dict2 = {key: list(lis1) for key in seq}
    # Printing created dict
    print("The newly created dict with list values : " + str(res_dict2) + Fore.LIGHTGREEN_EX + '\n# appending to lis1\n'
          + '>>lis1.append(4)\n# Printing dic after appending\n# Notice that append doesnt take place now.\n'
          + '>>print("The dict with list values after appending (no change) : ", str(res_dict2))')
    # appending to lis2
    lis1.append(4)
    # Printing dic after appending
    # Notice that append doesnt take place now.
    print("The dict with list values after appending (no change) : " + str(res_dict2) + Fore.LIGHTGREEN_EX
          + '\n# Python3 code to demonstrate\n# to initialize dictionary with list\n# using fromkeys()\n'
          + '# using fromkeys() to construct\n>>new_dict = dict.fromkeys(range(4), [])\n# printing result\n'
          + '>>print("New dictionary with empty lists as keys : " + str(new_dict))')
    # Python3 to demonstrate
    # to initialize dictionary with list
    # using fromkeys()
    # using fromkeys() to construct
    new_dict = dict.fromkeys(range(4), [])
    # printing result
    print("New dictionary with empty lists as keys : " + str(new_dict))


def from_bytes(bytes, byteorder='big', signed=False):
    if byteorder == 'little':
        little_ordered = list(bytes)
    elif byteorder == 'big':
        little_ordered = list(reversed(bytes))
    else:
        raise ValueError("byteorder must be either 'little' or 'big'")
    n = sum(b << i*8 for i, b in enumerate(little_ordered))
    if signed and little_ordered and (little_ordered[-1] & 0x80):
        n -= 1 << 8*len(little_ordered)
    return n


def describe_classmethod_int_from_bytes():
    print(Fore.YELLOW + 'classmethod int.from_bytes Method Test\n' + Fore.LIGHTGREEN_EX
          + '>>int.from_bytes(b\'\\x00\\x10\', byteorder=\'big\')\n' + Fore.WHITE
          + '{0}\n'.format(int.from_bytes(b'\x00\x10', byteorder='big')) + Fore.LIGHTGREEN_EX
          + '>>int.from_bytes(b\'\\x00\\x10\', byteorder=\'little\')\n' + Fore.WHITE
          + '{0}\n'.format(int.from_bytes(b'\x00\x10', byteorder='little')) + Fore.LIGHTGREEN_EX
          + '>>int.from_bytes(b\'\\xfc\\x00\', byteorder=\'big\', signed=True)\n' + Fore.WHITE
          + '{0}\n'.format(int.from_bytes(b'\xfc\x00', byteorder='big', signed=True)) + Fore.LIGHTGREEN_EX
          + '>>int.from_bytes(b\'\\xfc\\x00\', byteorder=\'big\', signed=False)\n' + Fore.WHITE
          + '{0}\n'.format(int.from_bytes(b'\xfc\x00', byteorder='big', signed=False)) + Fore.LIGHTGREEN_EX
          + '>>int.from_bytes([255, 0, 0], byteorder=\'big\')\n' + Fore.WHITE
          + '{0}\n'.format(int.from_bytes([255, 0, 0], byteorder='big')) + Fore.LIGHTGREEN_EX
          + '>>def from_bytes(bytes, byteorder=\'big\', signed=False):\n\tif byteorder == \'little\':\n\t\t'
          + 'little_ordered = list(bytes)\n\telif byteorder == \'big\':\n\t\tlittle_ordered = list(reversed(bytes))\n\t'
          + 'else:\n\t\traise ValueError("byteorder must be either \'little\' or \'big\'")\n\t'
          + 'n = sum(b << i*8 for i, b in enumerate(little_ordered))\n\t'
          + 'if signed and little_ordered and (little_ordered[-1] & 0x80):\n\t\tn -= 1 << 8*len(little_ordered)\n\t'
          + 'return n\n>>from_bytes(b\'\\xfc\\x00\', byteorder=\'big\', signed=True)\n' + Fore.WHITE
          + '{0}'.format(from_bytes(b'\xfc\x00', byteorder='big', signed=True)))


def describe_clear():
    print(Fore.YELLOW + 'clear Method Test\n\tFrom https://www.geeksforgeeks.org/python-list-clear-method/\n'
          + Fore.LIGHTGREEN_EX + '>>lis = [1, 2, 3]\n>>lis.clear()\n>>print(lis)')
    lis = [1, 2, 3]
    lis.clear()
    print(f'{lis}\n' + Fore.LIGHTGREEN_EX + '# creating a list\n>>numbers = [1, 2, 3, 4, 5]\n# using clear() function\n'
          + '>>numbers.clear()\n# printing numbers list\n>>print(numbers)')
    # creating a list
    numbers = [1, 2, 3, 4, 5]
    # using clear() function
    numbers.clear()
    # printing numbers list
    print(f'{numbers}\n' + Fore.LIGHTGREEN_EX + '# Python program to clear a list\n# using clear() method\n'
          + '# Creating list\n>>GEEK = [6, 0, 4, 1]\n>>print(\'GEEK before clear:\', GEEK)')
    # Python program to clear a list
    # using clear() method
    # Creating list
    GEEK = [6, 0, 4, 1]
    print(f'GEEK before clear: {GEEK}\n' + Fore.LIGHTGREEN_EX + '# Clearing list\n>>GEEK.clear()\n'
          + '>>print(\'GEEK after clear:\', GEEK)')
    # Clearing list
    GEEK.clear()
    print(f'GEEK after clear: {GEEK}\n' + Fore.LIGHTGREEN_EX + '>>my_list = [1, 2, 3, 4, 5]\n>>del my_list[:]\n'
          + '>>print(my_list)')
    my_list = [1, 2, 3, 4, 5]
    del my_list[:]
    print(f'{my_list}\n' + Fore.LIGHTGREEN_EX + '# Defining a 2-d list\n>>lis = [[0, 0, 1],\n\t[0, 1, 0],\n\t'
          + '[0, 1, 1]]\n# clearing the list\n>>lis.clear()\n>>print(lis)')
    # Defining a 2-d list
    lis = [[0, 0, 1],
           [0, 1, 0],
           [0, 1, 1]]
    # clearing the list
    lis.clear()
    print(f'{lis}\n' + Fore.LIGHTGREEN_EX + '# Defining a 2-d list\n>>lis = [[0, 0, 1],\n\t[0, 1, 0],\n\t[0, 1, 1]]\n'
          + '# clearing the list\n>>lis[0].clear()\n>>print(lis)')
    # Defining a 2-d list
    lis = [[0, 0, 1],
           [0, 1, 0],
           [0, 1, 1]]
    lis[0].clear()
    print(f'{lis}\n' + Fore.LIGHTGREEN_EX + '>>lis_1 = [1, 3, 5]\n>>lis_2 = [7, 9, 11]\n# using clear method\n'
          + '>>lis_1.clear()\n>>print("List after using clear():", lis_1)')
    lis_1 = [1, 3, 5]
    lis_2 = [7, 9, 11]
    # using clear method
    lis_1.clear()
    print(f'List after using clear(): {lis_1}\n' + Fore.LIGHTGREEN_EX + '# using del to clear items\ndel lis_2[:]\n'
          + '>>print("List after using del:", lis_2)')
    # using del to clear items
    del lis_2[:]
    print(f'List after using del: {lis_2}')


def f(x):
    return x


def fTwo(a, b, c):
    d = 40
    print(a, b, c, d)


def fThree(a, /, b, *, c, d):
    pass


def fFour(a, b, c = 10):
    x = 10
    y = 20


def fFive(b, a, c = 10):
    y = 10
    x = 20


a = 10


def foo(b, c, d):
    def bar():
        print(a, b, c, d)
        pass
    print(bar.__code__.co_names)


def foo_two():
    x = 10

    def bar_two():
        print(x)
    return bar_two


def describe_code_objects():
    print(Fore.YELLOW + 'Code Objects Test\n\tFrom https://www.tutorialspoint.com/python-code-objects\n\tand '
          + 'https://www.codeguage.com/courses/python/functions-code-objects\n' + Fore.LIGHTGREEN_EX
          + '>>code_string = """print("Hello Code Objects")"""\n# Create the code object\n'
          + '>>code_obj = compile(code_str, \'<string>\', \'exec\')\n# get the code object\n>>print(code_obj)')
    code_str = """print("Hello code Objects")"""
    # Create the code object
    code_obj = compile(code_str, '<string>', 'exec')
    # get the code object
    print(f'{code_obj}\n' + Fore.LIGHTGREEN_EX + '# Attributes of code object\n>>print(dir(code_obj))')
    # Attributes of code object
    print(f'{dir(code_obj)}\n' + Fore.LIGHTGREEN_EX + '# The filename\n>>print(code_obj.co_filename)')
    # The filename
    print(f'{code_obj.co_filename}\n' + Fore.LIGHTGREEN_EX + '# The first chunk of raw bytecode\n'
          + '>>print(code_obj.co_code)')
    # The first chunk of raw bytecode
    print(f'{code_obj.co_code}\n' + Fore.LIGHTGREEN_EX + '# The variable Names\n>>print(code_obj.co_varnames)')
    # The variable Names
    print(f'{code_obj.co_varnames}\n' + Fore.LIGHTGREEN_EX + '>>def f(x):\n\treturn x\n>>f.__code__\n' + Fore.WHITE
          + f'{f.__code__}\n' + Fore.LIGHTGREEN_EX + '>>def fTwo(a, b, c):\n\td = 40\n\tprint(a, b, c, d)\n'
          + '>>print(fTwo.__code__.co_nlocals)\n' + Fore.WHITE + f'{fTwo.__code__.co_nlocals}\n' + Fore.LIGHTGREEN_EX
          + '>>def fThree(a, /, b, *, c, d):\n\tpass\n>>fThree.__code__.co_argcount\n' + Fore.WHITE
          + f'{fThree.__code__.co_argcount}\n' + Fore.LIGHTGREEN_EX + '>>def fFour(a, b, c = 10):\n\tx = 10\n\ty = 20\n'
          + '>>fFour.__code__.co_varnames\n' + Fore.WHITE + f'{fFour.__code__.co_varnames}\n' + Fore.LIGHTGREEN_EX
          + '>>def fFive(b, a, c = 10):\n\ty = 10\n\tx = 20\n>>f.__code__.co_varnames\n' + Fore.WHITE
          + f'{fFive.__code__.co_varnames}\n' + Fore.LIGHTGREEN_EX + '>>a = 10\n>>def foo(b, c, d):\n\tdef bar():\n\t\t'
          + 'print(a, b, c, d)\n\t\tpass\n\tprint(bar.__code__.co_names)\n>>foo(1, 2, 3)')
    foo(1, 2, 3)
    print(Fore.LIGHTGREEN_EX + '>>def foo_two():\n\tx = 10\n\tdef bar_two():\n\t\tprint(x)\n\treturn bar_two\n'
          + '>>bar = foo_two()\n>>bar.__code__.co_freevars')
    bar = foo_two()
    print(f'{bar.__code__.co_freevars}\n' + Fore.LIGHTGREEN_EX + '>>foo_two.__code__.co_cellvars\n' + Fore.WHITE
          + f'{foo_two.__code__.co_cellvars}')


def describe_compile():
    print(Fore.YELLOW + 'Compile Test Method\n\tFrom https://www.geeksforgeeks.org/python-compile-function/\n'
          + Fore.LIGHTGREEN_EX + '>>srcCode = \'x = 10\\ny = 20\\nmul = x * y\\nprint("mul =", mul)\'\n'
          + '>>execCode = compile(srcCode, \'mulstring\', \'exec\')\n>>exec(execCode)')
    srcCode = 'x = 10\ny = 20\nmul = x * y\nprint("mul =", mul)'
    execCode = compile(srcCode, 'mulstring', 'exec')
    exec(execCode)
    print(Fore.LIGHTGREEN_EX + '>>x = 50\n>>a = compile(\'x\', \'test\', \'single\')\n>>print(type(a))')
    x = 50
    a = compile('x', 'test', 'single')
    print(str(type(a)) + Fore.LIGHTGREEN_EX + '\n>>exec(a)')
    exec(a)
    print(Fore.BLUE + '# compile_main file\nString = "Welcome to Geeksforgeeks"\nprint(String)\n' + Fore.LIGHTGREEN_EX
          + '>>f = open(\'compile_main.py\', \'r\')\n>>temp = f.read()\n>>f.close()\n'
          + '>>code = compile(temp, \'compile_main.py\', \'exec\')\n>>exec(code)')
    f = open('compile_main.py', 'r')
    temp = f.read()
    f.close()
    code = compile(temp, 'compile_main.py', 'exec')
    exec(code)
    print(Fore.LIGHTGREEN_EX + '>>x = 50\n# Note eval is used for statement\n'
          + '>>a = compile(\'x == 50\', \'\', \'eval\')\n>>print(eval(a))')
    # Note eval is used for statement
    a = compile('x == 50', '', 'eval')
    print(eval(a))


def describe_iter():
    print(Fore.YELLOW + 'container.__iter__() Method Test\n\t'
          + 'From https://www.geeksforgeeks.org/python-__iter__-__next__-converting-object-iterator/\n'
          + Fore.LIGHTGREEN_EX + '# Python code demonstrating\n# basic use of iter()\n'
          + '>>listA = [\'a\', \'e\', \'i\', \'o\', \'u\']\n>>iter_listA = iter(listA)\n>>try:\n\t'
          + 'print(next(iter_listA))\n\tprint(next(iter_listA))\n\tprint(next(iter_listA))\n\t'
          + 'print(next(iter_listA))\n\tprint(next(iter_listA))\n\tprint(next(iter_listA)) # StopIteration Error\n'
          + 'except:\n\tpass')
    listA = ['a', 'e', 'i', 'o', 'u']
    iter_listA = iter(listA)
    try:
        print(next(iter_listA))
        print(next(iter_listA))
        print(next(iter_listA))
        print(next(iter_listA))
        print(next(iter_listA))
        print(next(iter_listA))
    except:
        pass
    print(Fore.LIGHTGREEN_EX + '# Python code demonstrating\n# basic use of iter()\n>>lst = [11, 22, 33, 44, 55]\n'
          + '>>iter_lst = iter(lst)\n>>while True:\n\ttry:\n\t\tprint(iter_lst.next())\n\texcept:\n\t\tbreak')
    # Python code demonstrating
    # basic use of iter()
    lst = [11, 22, 33, 44, 55]
    iter_lst = iter(lst)
    while True:
        try:
            print(iter_lst.__next__())
        except:
            break
    print(Fore.LIGHTGREEN_EX + '# Python code demonstrating\n# basic use of iter()\n'
          + '>>listB = [\'Cat\', \'Bat\', \'Sat\', \'Mat\']\n>>iter_listB = listB.__iter__()\ntry:\n\t'
          + 'print(iter_listB.__next__())\n\tprint(iter_listB.__next__())\n\tprint(iter_listB.__next__())\n\t'
          + 'print(iter_listB.__next__())\n\tprint(iter_listB.__next__())\nexcept:\n\t'
          + 'print(" \\nThrowing \'StopIterationError\'", "I cannot count more.")')
    # Python code demonstrating
    # basic use of iter()
    listB = ['Cat', 'Bat', 'Sat', 'Mat']
    iter_listB = listB.__iter__()
    try:
        print(iter_listB.__next__())
        print(iter_listB.__next__())
        print(iter_listB.__next__())
        print(iter_listB.__next__())
        print(iter_listB.__next__())
    except:
        print(" \nThrowing 'StopIterationError'", "I cannot count more.")
    print(Fore.LIGHTGREEN_EX + '>>class IterCounter:\n\tdef __init__(self, start, end):\n\t\tself.num = start\n\t\t'
          + 'self.end = end\n\n\tdef __iter__(self):\n\t\treturn self\n\n\tdef __next__(self):\n\t\t'
          + 'if self.num > self.end:\n\t\t\traise StopIteration\n\t\telse:\n\t\t\t'
          + 'self.num += 1\n\t\t\treturn self.num - 1\n# Driver code\n>>a, b = 2, 5\n>>c1 = IterCounter(a, b)\n'
          + '>>c2 = IterCounter(a, b)\n# Way 1-to print the range without iter()\n'
          + '>>print("Print the range without iter()")')
    a, b = 2, 5
    c1 = IterCounter(a, b)
    c2 = IterCounter(a, b)
    # Way 1-to print the range without iter()
    print("Print the range without iter()\n" + Fore.LIGHTGREEN_EX + '>>for i in c1:\n\t'
          + 'print("Eating more Pizzas, counting ", i, end="\\n")')
    for i in c1:
        print("Eating more Pizzas, counting ", i, end="\n")
    print(Fore.LIGHTGREEN_EX + 'print("\\nPrint the range using iter()\\n")' + Fore.WHITE
          + '\nPrint the range using iter()\n' + Fore.LIGHTGREEN_EX
          + '# Way 2 - using iter()\n>>obj = iter(c2)\n>>try:\n\t'
          + 'while True: # Print till error raised\n\t\tprint("Eating more Pizzas, counting ", next(obj))\n'
          + '  except:\n\t# when StopIteration raised, Print custom message\n\tprint("\nDead on overfood, GAME OVER")')
    # Way 2 - using iter()
    obj = iter(c2)
    try:
        while True: # Print till error raised
            print("Eating more Pizzas, counting ", next(obj))
    except:
        # When StopIteration raised, Print custom message
        print("\nDead on overfood, GAME OVER")
    print(Fore.LIGHTGREEN_EX + '>>class MyRange:\n\tdef __init__(self, start, end):\n\t\tself.current = start\n\t\t'
          + 'self.end = end\n\n\tdef __iter__(self):\n\t\treturn self\n\n\tdef __next__(self):\n\t\t'
          + 'if self.current >= self.end:\n\t\t\traise StopIteration\n\t\tcurrent = self.current\n\t\tself.current += 1'
          + '\n\t\treturn current\n# Example usage\nmy_range = MyRange(1, 5)\n>>for num in my_range:\n\t'
          + 'print(num) # Output: 1, 2, 3, 4')
    # Example usage
    my_range = MyRange(1, 5)
    for num in my_range:
        print(num) # Output: 1, 2, 3, 4


def describe_contextmanager_enter():
    print(Fore.YELLOW + 'contextmanager.__enter__() Method Test\n\t'
          + 'From https://www.pythonmorsels.com/creating-a-context-manager/#a-useful-context-manager\n'
          + Fore.LIGHTGREEN_EX + '>>with open("example.txt", "w") as file:\n\tprint(file.write("Hello, world!"))\n')
    with open("example.txt", "w") as file:
        print(file.write("Hello, world!"))
    print(Fore.LIGHTGREEN_EX + '>>file.closed\n' + Fore.WHITE + f'{file.closed}\n' + Fore.LIGHTGREEN_EX
          + '>>class Context_Manager_Example:\n\tdef __enter__(self):\n\t\tprint("enter")\n\n\t'
          + 'def __exit__(self, exc_type, exc_val, exc_tb):\n\t\tprint("exit")\n>>with Context_Manager_Example():\n\t'
          + 'print("Yay Python!")')
    with Context_Manager_Example():
        print("Yay Python!")
    print(Fore.LIGHTGREEN_EX + '>>import os\n>>class set_env_var:\n\tdef __init__(self, var_name, new_value):\n\t\t'
          + 'self.var_name = var_name\n\t\tself.new_value = new_value\n\n\tdef __enter__(self):\n\t\t'
          + 'self.original_value = os.environ.get(self.var_name)\n\t\tos.environ[self.var_name] = self.new_value\n\n\t'
          + 'def __exit__(self, exc type, excval, exc_tb)\n\t\tif self.original_value is None:\n\t\t\t'
          + 'del os.environ[self.var_name]\n\t\telse:\n\t\t\tos.environ[self.var_name] = self.original_value\n'
          + '>>try:\n\tprint("USER env var is", os.environ["USER"])\n  except KeyError as e:\n\t'
          + 'print(traceback.format_exc())')
    try:
        print("USER env var is", os.environ["USER"])
    except KeyError as e:
        print(traceback.format_exc())
    print(Fore.LIGHTGREEN_EX + '>>with set_env_var("USER", "akin"):\n\tprint("USER env var is", os.environ["USER"])')
    with set_env_var("USER", "akin"):
        print("USER env var is", os.environ["USER"])
    print(Fore.LIGHTGREEN_EX + '>>try:\n\tprint("USER env var is", os.environ["USER"])\n  except KeyError as e:\n\t'
          + 'print(traceback.format_exc())')
    try:
        print("USER env var is", os.environ["USER"])
    except KeyError as e:
        print(traceback.format_exc())
    print(Fore.LIGHTGREEN_EX + '>>with set_env_var("USER", "akin") as result:\n\t'
          + 'print("USER env var is", os.environ["USER"])\n\tprint("Result from __enter__ method:", result)')
    with set_env_var("USER", "akin") as result:
        print("USER env var is", os.environ["USER"])
        print("Result from __enter__ method:", result)
    print(Fore.LIGHTGREEN_EX + '>>import time\nclass Timer:\n\tdef __enter__(self):\n\t\t'
          + 'self.start = time.perf_counter()\n\t\treturn self\n\n\tdef __exit__(self, exc_type, exc_val, exc_tb):'
          + '\n\t\tself.stop = time.perf_counter()\n\t\tself.elapsed = self.stop - self.start\n>>t = Timer()\n'
          + '>>with t:\n\tresult = sum(range(10_000_000))\n>>t.elapsed')
    t = Timer()
    with t:
        result = sum(range(10_000_000))
    print(f'{t.elapsed}\n' + Fore.LIGHTGREEN_EX + '>>with Timer() as t:\n\tresult = sum(range(10_000_000))\n'
          + '>>t.elapsed')
    with Timer() as t:
        result = sum(range(10_000_000))
    print(t.elapsed)


@contextmanager
def set_env_var_func(var_name, new_value):
    original_value = os.environ.get(var_name)
    os.environ[var_name] = new_value
    try:
        yield
    finally:
        if original_value is None:
            del os.environ[var_name]
        else:
            os.environ[var_name] = original_value


def describe_contextmanager_exit():
    print(Fore.YELLOW + 'contextmanager.__enter__() Method Test\n\t'
          + 'From https://www.pythonmorsels.com/creating-a-context-manager/#a-useful-context-manager\n'
          + Fore.LIGHTGREEN_EX + '>>import logging\n'
          + Fore.BLUE + 'class LogException:\n\tdef __init__(self, logger, level=logging.ERROR, suppress=False):\n\t\t'
          + 'self.logger, self.level,self.suppress = logger, level, suppress\n\n\tdef __enter__(self):\n\t\treturn self'
          + '\n\n\tdef __exit__(self, exc_type, exc_value, exc_tb):\n\t\tif exc_type is not None:\n\t\t\t'
          + 'info = (exc_type, exc_val, exc_tb):\n\t\t\t'
          + 'self.logger.log(self.level, "Exception occurred", exc_info=info)\n\t\t\treturn self.suppress\n\t\t'
          + 'return False\n' + Fore.LIGHTGREEN_EX + '>>import logging\n>>from log_exception import LogException\n'
          + '>>logging.basicConfig(level = logging.DEBUG)\n>>logger.getLogger("example")\n>>with LogException(logger):'
          + '\n\tresult = 1 / 0 # This will cause a ZeroDivisionError\n>>print("That\'s the end of our program")\n'
          + '# Actual code\n>>try:\n\twith LogException(logger):\n\t\t'
          + 'result = 1 / 0 # This will cause a ZeroDivisionError\n  except ZeroDivisionError as e:\n\t'
          + 'print("This is where \'Thats\'s the end of our program\' fails to print!")')
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("example")
    try:
        with LogException(logger):
            result = 1 / 0 # This will cause a ZeroDivisionError
    except ZeroDivisionError as e:
        print("This is where 'That's the end of our program' fails to print!")
    print(Fore.LIGHTGREEN_EX + '>>with LogException(logger, suppress=True):\n\t'
          + 'result = 1 / 0 # This will cause a ZeroDivisionError\n>>print("That\'s the end of our program")')
    with LogException(logger, suppress=True):
        result = 1 / 0 # This will cause a ZeroDivisionError
    print("That's the end of our program\n" + Fore.LIGHTGREEN_EX + '>>from contextlib import contextmanager\n'
          + '>>import os\n>>@contextmanager\ndef set_env_var_func(var_name, new_value):\n\t'
          + 'original_value = os.environ.get(var_name)\n\tos.environ[var_name] = new_value\n\ttry:\n\t\tyield\n\t'
          + 'finally:\n\t\tif original_value is None:\n\t\t\tdel os.environ[var_name]\n\t\telse:\n\t\t\t'
          + 'os.environ[var_name] = original_value\n>>with set_env_var_func("COIN", "dime"):\n\t'
          + 'print("COIN env var is", os.environ["COIN"])')
    with set_env_var_func("COIN", "dime"):
        print("COIN env var is", os.environ["COIN"])


def describe_conversion_flag_characters():
    print(Fore.YELLOW + 'Conversion Flag Characters Test\n' + Fore.LIGHTGREEN_EX
          + '>>print(\'%(number)#o\\n%(moon)#x\\n%(orange)#X\\n%(fire)#e\\n%(fire)#.3e\'\n % '
          + '{\'number\': 25, \'moon\': 57, \'orange\': 42, \'fire\': 14})\n'
          + Fore.WHITE + '%(number)#o\n%(moon)#x\n%(orange)#X\n%(fire)#e\n%(fire)#.3e\n'
          % {'number': 25, 'moon': 59, 'orange': 42, 'fire': 14} + Fore.LIGHTGREEN_EX
          + '>>print(\'%(another)08d\\n%(fly)08f\' % {\'another\': 9, \'fly\': 4})\n' + Fore.WHITE
          + '%(another)08d\n%(fly)08f\n' % {'another': 9, 'fly': 4} + Fore.LIGHTGREEN_EX
          + '>>print(\'%(glue)-8d\\n%(bag)-08d\' % {\'glue\': 6, \'bag\': 12})\n' + Fore.WHITE
          + '%(glue)-8d\n%(bag)-08d\n' % {'glue': 6, 'bag': 12} + Fore.LIGHTGREEN_EX
          + '>>print(\'%(home) 20s\' % {\'home\': \'Sioux Falls\'})\n' + Fore.WHITE
          + '%(home) 20s\n' % {'home': 'Sioux Falls'} + Fore.LIGHTGREEN_EX
          + '>>print(\'%(young)++6i%(old)%(old)+-6i\' % {\'young\': 16}, {\'old\': 38})\n' + Fore.WHITE
          + '%(young)++6i\n%(old)+-6i' % {'young': -16, 'old': 38})


def describe_conversion_specifier_order_in_print():
    print(Fore.YELLOW + 'Conversion Specifier Order In Print Test\n' + Fore.LIGHTGREEN_EX
          + '>>print(b\'%(language)s has %(number)03d quote types.\' % {b\'language\': b"Python", b"number": 2})')
    print(b'%(language)s has %(number)03d quote types.' % {b'language': b'Python', b'number': 2})
    print(Fore.LIGHTGREEN_EX + '>>print(\'%(prime)+-15.3le\' % {\'prime\': 1009})\n' + Fore.WHITE
          + '%(prime)++15.3le' % {'prime': 1009})


def describe_conversion_types():
    print(Fore.YELLOW + 'Conversion Types Test\n' + Fore.LIGHTGREEN_EX + '>>print(\'%(signed)d\' % {\'signed\': 5})\n'
          + Fore.WHITE + '%(signed)d\n' % {'signed': 5} + Fore.LIGHTGREEN_EX
          + '>>print(\'%(integer)i\' % {\'integer\': 6})\n' + Fore.WHITE + '%(integer)i\n' % {'integer': 6}
          + Fore.LIGHTGREEN_EX + '>>print(\'%(octal)o\\n%(octal)#o\' % {\'octal\': 12})\n' + Fore.WHITE
          + '%(octal)o\n%(octal)#o\n' % {'octal': 12} + Fore.LIGHTGREEN_EX
          + '>>print(\'%(obsolete)u\\n%(obsolete)#u\' % {\'obsolete\': 13})\n' + Fore.WHITE
          + '%(obsolete)u\n%(obsolete)#u\n' % {'obsolete': -13} + Fore.LIGHTGREEN_EX
          + '>>print(\'%(hex)x\\n%(hex)#x\' % {\'hex\': 15})\n' + Fore.WHITE + '%(hex)x\n%(hex)#x\n' % {'hex': 15}
          + Fore.LIGHTGREEN_EX + '>>print(\'%(hexcap)X\\n%(hexcap)#X\' % {\'hexcap\': 28})\n' + Fore.WHITE
          + '%(hexcap)X\n%(hexcap)#X\n' % {'hexcap': 28} + Fore.LIGHTGREEN_EX
          + '>>print(\'%(exp)e\\n%(exp)#e\' % {\'exp\': 1456})\n' + Fore.WHITE + '%(exp)e\n%(exp)#e\n' % {'exp': 1456}
          + Fore.LIGHTGREEN_EX + '>>print(\'%(expcap)E\\n%(expcap)#E\' % {\'expcap\': 3425})\n' + Fore.WHITE
          + '%(expcap)E\n%(expcap)#E\n' % {'expcap': 3425} + Fore.LIGHTGREEN_EX
          + '>>print(\'%(float)f\\n%(float)#f\' % {\'float\': 68.78})\n' + Fore.WHITE + '%(float)f\n%(float)#f\n'
          % {'float': 68.78} + Fore.LIGHTGREEN_EX + '>>print(\'%(floatsix)F\\n%(floatsix)#F\' % {\'floatsix\': 34.9})\n'
          + Fore.WHITE + '%(floatsix)F\n%(floatsix)#F\n' % {'floatsix': 34.9} + Fore.LIGHTGREEN_EX
          + '>>print(\'%(exponent)g\\n%(exponent)#g\\n%(preck).3g\' % {\'exponent\': 0.00008, \'preck\': 12345})\n'
          + Fore.WHITE + '%(exponent)g\n%(exponent)#g\n%(preck).3g\n' % {'exponent': 0.00008, 'preck': 12345}
          + Fore.LIGHTGREEN_EX
          + '>>print(\'%(exponcap)G\\n%(exponcap)#G\\n%(preckcap).3G\' % {\'exponcap\': 0.00008, \'preckcap\': 12345})'
          + Fore.WHITE + '\n%(exponcap)G\n%(exponcap)#G\n%(preckcap).3G\n' % {'exponcap': 0.00008, 'preckcap': 12345}
          + Fore.LIGHTGREEN_EX + '>>print(\'%(character)c\' % {\'character\': \'r\'})\n' + Fore.WHITE
          + '%(character)c\n' % {'character': 'r'} + Fore.LIGHTGREEN_EX
          + '>>print(\'%(string)r\\n%(string)#r\' % {\'string\': \'Hello World\'})\n' + Fore.WHITE
          + '%(string)r\n%(string)#r\n' % {'string': 'Hello World'} + Fore.LIGHTGREEN_EX
          + '>>print(\'%(styring)s\\n%(styring)#s\' % {\'styring\': \'Help Me?\'})\n' + Fore.WHITE
          + '%(styring)s\n%(styring)#s' % {'styring': 'Help Me?'})


def describe_copy():
    print(Fore.YELLOW + 'copy() Method Test\nFrom https://www.geeksforgeeks.org/python-list-copy-method/\n'
          + Fore.LIGHTGREEN_EX + '# Using list fruits\n>>fruits = ["mango", "apple", "strawberry"]\n'
          + '# creating a copy shakes\n>>shakes = fruits.copy()\n# printing shakes list\n>>print(shakes)')
    # Using list fruits
    fruits = ["mango", "apple", "strawberry"]
    # creating a copy shakes
    shakes = fruits.copy()
    # printing shakes list
    print(str(shakes) + Fore.LIGHTGREEN_EX + '\n# Using list girls\n>>girls = ["Priya", "Neha", "Radha", "Nami"]\n'
          + '# Creating new copy\n>>girlstudent = girls.copy()\n# printing new list\n>>print(girlstudent)')
    # Using list girls
    girls = ["Priya", "Neha", "Radha", "Nami"]
    # Creating new copy
    girlstudent = girls.copy()
    # printing new list
    print(str(girlstudent) + Fore.LIGHTGREEN_EX + '\n>>lis = [\'Geeks\', \'for\', \'Geeks\']\n>>new_list = lis.copy()\n'
          + '>>print(\'Copied List:\', new_list)')
    lis = ['Geeks', 'for', 'Geeks']
    new_list = lis.copy()
    print('Copied List: ' + str(new_list) + Fore.LIGHTGREEN_EX + '\n# Initializing list\n>>lis1 = [1, 2, 3, 4]\n'
          + '# Using copy() to create a shallow copy\n>>lis2 = lis1.copy()\n# Printing new list\n'
          + '>>print("The new list created is : " + str(lis2))')
    # Initializing list
    lis1 = [1, 2, 3, 4]
    # Using copy() to create a shallow copy
    lis2 = lis1.copy()
    # Printing new list
    print("The new list created is : " + str(lis2) + Fore.LIGHTGREEN_EX
          + '\n# Adding new element to new list\n>>lis2.append(5)\n'
          + '# Printing lists after adding new element\n# No change in old list\n'
          + '>>print("The new list after adding new element : " + str(lis2))')
    # Adding new element to new list
    lis2.append(5)
    # Printing lists after adding new element
    # No change in old list
    print("The new list after new element : " + str(lis2) + Fore.LIGHTGREEN_EX
          + '\n>>print("The old list after adding new element to new list : " + str(lis1))\n' + Fore.WHITE
          + 'The old list after adding new element to new list ' + str(lis1) + Fore.LIGHTGREEN_EX + '\n>>import copy\n'
          + '# Initializing list\n>>list1 = [1, [2, 3], 4]\n>>print("list 1 before modification:\\n", list1)')
    # Initializing list
    list1 = [1, [2, 3], 4]
    print('list 1 before modification ' + str(list1) + Fore.LIGHTGREEN_EX + '\n# all changes are reflected\n'
          + '>>list2 = list1\n# shallow copy - changes to\n# nested list is reflected,\n'
          + '# same as copy.copy(), slicing\n>>list3 = list1.copy()\n# deep copy - no change is reflected\n'
          + '>>list4 = copy.deepcopy(list1)\n>>list1.append(5)\n>>list1[1][1] = 999\n'
          + '>>print("list 1 after modification:\\n", list1)')
    # all changes are reflected
    list2 = list1
    # shallow copy - changes to
    # nested list is reflected,
    # same as copy.copy(), slicing
    list3 = list1.copy()
    # deep copy - no change is reflected
    list4 = copy.deepcopy(list1)
    list1.append(5)
    list1[1][1] = 999
    print(f'list 1 after modification:\n {list1}\n' + Fore.LIGHTGREEN_EX
          + '>>print("list 2 after modification:\\n", list2)\n' + Fore.WHITE + f'list 2 after modification:\n {list2}\n'
          + Fore.LIGHTGREEN_EX + '>>print("list 3 after modification:\\n", list3)\n' + Fore.WHITE
          + f'list 3 after modification:\n {list3}\n' + Fore.LIGHTGREEN_EX
          + '>>print("list 4 after modification:\\n", list4)\n' + Fore.WHITE + f'list 4 after modification:\n {list4}\n'
          + Fore.LIGHTGREEN_EX + '>>list = [2, 4, 6]\n>>new_list = list[:]\n>>new_list.append(\'a\')\n'
          + '>>print(\'Old List:\', list)')
    list = [2, 4, 6]
    new_list = list[:]
    new_list.append('a')
    print(f'Old List: {list}\n' + Fore.LIGHTGREEN_EX + '>>print(\'New List:\', new_list)\n' + Fore.WHITE
          + f'New List: {new_list}')


def describe_copyright_and_credits():
    print(Fore.YELLOW + 'Copyright and Credits Test\n' + Fore.LIGHTGREEN_EX + '>>class ExperimentCopyright:\n\t'
          + '__author__ = \'Clayton Buus\'\n\t__copyright__ = f\'Copyright {chr(169)} 2024 Clayton Buus.\\n'
          + 'All Rights Reserved.\n\t__credit__ = \'Thanks to GeeksForGeeks, Programiz, '
          + 'and W3Schools for providing the ideas for\\n\' /\n\t\'most of the code outside of this class.\'\n\t'
          + '__license__ = \'Public Domain\'\n\t__version__ = \'1.0\'\n\n\tdef __init__(self):\n\t\t'
          + 'print(f\'{ExperimentCopyright.__copyright__}\n{ExperimentCopyright.__credit__}\')'
          + '\n>>ExperimentCopyright()')
    ExperimentCopyright()


async def main_coroutine():
    print('hello')
    await asyncio.sleep(1)
    print('world')


async def say_after(delay, what):
    await asyncio.sleep(delay)
    print(what)


async def main_time():
    print(f"started at {time.strftime('%X')}")
    await say_after(1, 'hello')
    await say_after(2, 'world')
    print(f"finished at {time.strftime('%X')}")


async def main_task():
    task1 = asyncio.create_task(say_after(1, 'hello'))
    task2 = asyncio.create_task(say_after(2, 'world'))
    print(f"started at {time.strftime('%X')}")
    # Wait until both tasks are completed (should take
    # around 2 seconds.)
    await task1
    await task2
    print(f"finished at {time.strftime('%X')}")


async def main_implicit():
    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(say_after(1, 'hello'))
        task2 = tg.create_task(say_after(2, 'world'))
        print(f"started at {time.strftime('%X')}")
    # The await is implicit when the context manager exits.
    print(f"finished at {time.strftime('%X')}")


def describe_coroutines():
    print(Fore.YELLOW + 'Coroutines Test Method\n\tFrom https://docs.python.org/3/library/asyncio-task.html\n'
          + Fore.LIGHTGREEN_EX + '>>import asyncio\n>>async def main_coroutine():\n\tprint(\'hello\')\n\t'
          + 'await asyncio.sleep(1)\n\tprint(\'world\')\n>>asyncio.run(main_coroutine())')
    asyncio.run(main_coroutine())
    print(Fore.LIGHTGREEN_EX + '>>import time\n>>async def say_after(delay, what):\n\tawait asyncio.sleep(delay)\n\t'
          + 'print(what)\n>>async def main_time():\n\tprint(f"started at {time.strftime(\'%X\')}")\n\t'
          + 'await say_after(1, \'hello\')\n\tawait say_after(2, \'world\')\n\t'
          + 'print(f"finished at {time.strftime(\'%X\')}")\n>>asyncio.run(main_time())')
    asyncio.run(main_time())
    print(Fore.LIGHTGREEN_EX + '>>async def main_task():\n\ttask1 = asyncio.creat_task(say_after(1, \'hello\'))\n\t'
          + 'task2 = asyncio.create_task(say_after(2, \'world\'))\n\tprint(f"started at {time.strftime(\'%X\')}")\n\t'
          + '# Wait until both tasks are completed (should take\n\t# around 2 seconds.)\n\tawait task1\n\tawait task2'
          + '\n\tprint(f"finished at {time.strftime(\'%X\')}")\n>>asyncio.run(main_task())')
    asyncio.run(main_task())
    print(Fore.LIGHTGREEN_EX + '>>async def main_implicit():\n\tasync with asyncio.TaskGroup() as tg:\n\t\t'
          + 'task1 = tg.create_task(say_after(1, \'hello\'))\n\t\t'
          + 'task2 = tg.create_task(say_after(2, \'world\'))\n\t\tprint(f"started at {time.strftime(\'%X\')}")\n\t'
          + '# The await is implicit when the context manager exits.\n\tprint(f"finished at {time.strftime(\'%X\')}")\n'
          + '>>asyncio.run(main_implicit())')
    asyncio.run(main_implicit())


def describe_d_or_other():
    print(Fore.YELLOW + 'd | other Test\n' + Fore.LIGHTGREEN_EX
          + '>>dOne = {\'apples\': 3, \'oranges\': 5, \'banana\': 9, \'pomegranate\': 14, \'cherry\': 2}\n'
            '>>dTwo = {\'apples\': 4, \'oranges\': 5, \'banana\': 7, \'grapes\': 6, \'raspbery\': 12}\n'
            '>>dOne | dTwo')
    dOne = {'apples': 3, 'oranges': 5, 'banana': 9, 'pomegranate': 14, 'cherry': 2}
    dTwo = {'apples': 4, 'oranges': 5, 'banana': 7, 'grapes': 6, 'raspberry': 12}
    print(f'{dOne | dTwo}\n' + Fore.LIGHTGREEN_EX + '>>dTwo | dOne\n' + Fore.WHITE + f'{dTwo | dOne}')


def describe_d_or_assign_other():
    print(Fore.YELLOW + 'd |= other Test\n' + Fore.LIGHTGREEN_EX
          + '>>pOne = {\'Lean Meats\': \'Kangaroo\', \'Poultry\': \'Emu\', \'Nuts and seeds\': \'macadamia\'}\n'
            '>>pTwo = {\'Lean Meats\': \'Kangaroo\', \'Poultry\': \'Goose\', \'Legumes/Beans\': \'lentils\'}\n'
            '>>pOneCopy = pOne.copy()\n>>pOne |= pTwo')
    pOne = {'Lean Meats': 'Kangaroo', 'Poultry': 'Emu', 'Nuts and seeds': 'macadamia'}
    pTwo = {'Lean Meats': 'Kangaroo', 'Poultry': 'Goose', 'Legumes/Beans': 'lentils'}
    pOneCopy = pOne.copy()
    pOne |= pTwo
    print(f'{pOne}\n' + Fore.LIGHTGREEN_EX + '>>pOne\n' + Fore.WHITE + f'{pOne}\n' + Fore.LIGHTGREEN_EX
          + '>>pTwo |= pOneCopy # <- this is the original pOne')
    pTwo |= pOneCopy
    print(f'{pTwo}\n' + Fore.LIGHTGREEN_EX + '>>pTwo\n' + Fore.WHITE + f'{pTwo}')


def describe_d_key():
    print(Fore.YELLOW + 'd[key] Test\n' + Fore.LIGHTGREEN_EX
          + '>>d = {\'Mission To Mars\': \'LEGO MX-71 Recon Dropship\', \'Nasa\': \'Mars Rover Perseverance 42158\'}\n'
            '>>d[\'Nasa\']')
    d = {'Mission To Mars': 'LEGO MX-71 Recon Dropship', 'Nasa': 'Mars Rover Perseverance 42158'}
    print('{0}\n'.format(d['Nasa']) + Fore.LIGHTGREEN_EX + '>>try:\n\td[\'Star Wars\']\n  '
          + 'except KeyError as e:\n\tprint(traceback.format_exc())')
    try:
        d['Star Wars']
    except KeyError as e:
        print(traceback.format_exc())
    print(Fore.LIGHTGREEN_EX + '>>class InputCounter(dict):\n\tdef __missing__(self, key):\n\t\treturn 0\n'
                               '>>c = Counter()\n>>c[\'red\']')
    c = Counter()
    print(str(c['red']) + Fore.LIGHTGREEN_EX + '\n>>c[\'red\'] += 1\n>>c[\'red\']')
    c['red'] += 1
    print(str(c['red']))


def describe_add_to_dictionary():
    print(Fore.YELLOW + 'd[key] = value Test\n' + Fore.LIGHTGREEN_EX + '>>home = {\'remote\': \'visio\'}\n'
          '>>home[\'body armor\'] = \'peach mago\'>>home[\'body armor\']')
    home = {'remote': 'visio', 'body armor': 'peach mango'}
    print(str(home['body armor']) + Fore.LIGHTGREEN_EX + '\n>>home[\'remote\'] = \'netflix\'\n>>home[\'remote\']')
    home['remote'] = 'netflix'
    print(home['remote'])


# initializing function
def gfg():
    return "You just called for success !!"


def Geekforgeeks():
    pass


# initializing function
def get_function_name():
    # get the frame object of the function
    frame = inspect.currentframe()
    return frame.f_code.co_name


# Python program showing
# how to invoke descriptor
def __getattribute__(self, key):
    v = object.__getattribute__(self, key)
    if hasattr(v, '__get__'):
        return v.__get__(None, self)
    return v



# A generator function that yields 1 for first time,
# 2 second time and 3 third time
def simpleGeneratorFun():
    yield 1
    yield 2
    yield 3


def fib(limit):
    a, b = 0, 1
    while b < limit:
        yield b
        a, b = b, a + b


def describe_definition_p__name__():
    print(Fore.YELLOW
          + 'definition.__name__ Test\n\t'
            'From https://www.geeksforgeeks.org/python-program-to-get-the-class-name-of-an-instance/\n\t'
            'https://www.geeksforgeeks.org/python-how-to-get-function-name/\n\t'
            'https://www.geeksforgeeks.org/descriptor-in-python/\n\t'
            'https://www.geeksforgeeks.org/generators-in-python/\n'
          + Fore.CYAN + 'Class.__name__\n' + Fore.LIGHTGREEN_EX
          + '# this is a class named car\n>>class car:\n\tdef parts(self):\n\t\tpass\n>>c = car()\n'
            '# prints the class of the object c\n>>print(c.__class__)')
    c = Car()
    # prints the class of the object c
    print(str(c.__class__) + Fore.LIGHTGREEN_EX
          + '\n# this prints the name of the class\n>>classes = c.__class__\n# prints the name of the class\n'
            '>>print(classes.__name__)')
    # this prints the name of the class
    classes = c.__class__
    # prints the name of the class
    print(str(classes.__name__) + Fore.LIGHTGREEN_EX + '\n# this prints the class of a c\n>>print(type(c).__name__)\n'
          + Fore.WHITE + f'{type(c).__name__}\n' + Fore.LIGHTGREEN_EX
          + '# class for getting the class name\n>>class test:\n\t@property\n\tdef cls(self):\n\t\t'
            'return type(self).__name__\n>>a = test()\n>>print(a.cls)')
    a = Test()
    print(str(a.cls) + Fore.LIGHTGREEN_EX
          + '\n>>class Bike:\n\tdef __init__(self, name, b):\n\t\tself.name = name\n\t\tself.car = self.Car(b)\n\n\t'
            'class Car:\n\t\tdef __init__(self, car):\n\t\t\tself.car = car\n'
            '>>vehicle = bike("orange", [\'patatoes\'])\n>>print(vehicle.car.__class__.__name__)')
    vehicle = Bike("orange", ['potatoes'])
    print(f'{vehicle.car.__class__.__name__}\n' + Fore.LIGHTGREEN_EX + '>>print(vehicle.car.__class__.__qualname__)\n'
          + Fore.WHITE + f'{vehicle.car.__class__.__qualname__}\n' + Fore.LIGHTGREEN_EX
          + '>>import inspect\n>>class Empty:\n\tpass\n>>obj = Empty()\n>>members = inspect.getmembers(obj)\n'
            '>>class_name = [m[1] for m in members if m[0] == \'__class__\'][0]\n'
            '>>print(class_name.__name__) # Output: \'MyClass\'')
    obj = Empty()
    members = inspect.getmembers(obj)
    class_name = [m[1] for m in members if m[0] == '__class__'][0]
    print(f'{class_name.__name__}\n' + Fore.LIGHTGREEN_EX
          + '>>class MyNewBaseClass:\n\tdef __init_subclass__(cls, **kwargs):\n\t\t'
            'super().__init_subclass__(**kwargs)\n\t\tcls.name = cls.__name__\n>>class MyNewClass(MyNewBaseClass):\n\t'
            'pass\n>>obj = MyNewClass()\n>>print(obj.name)')
    obj = MyNewClass()
    print(f'{obj.name}\n' + Fore.CYAN + 'Function.__name__\n' + Fore.LIGHTGREEN_EX
          + '# initializing function\ndef gfg():\n\treturn "You just called for success !!"\n# printing function name\n'
            '# using function.__name__\nprint("The name of function is : " + gfg.__name__)')
    # printing function name
    # using function.__name__
    print("The name of function is : " + gfg.__name__ + Fore.LIGHTGREEN_EX
          + '\ndef Geekforgeeks():\n\tpass\nclass Geekfgeeks(object):\n\tdef my_method(self):\n\t\tpass\n'
            '# "my_function"\n>>print(Geekforgeeks.__qualname__)')
    # "my_function"
    print(Geekforgeeks.__qualname__ + Fore.LIGHTGREEN_EX
          + '\n# "My_Class.my_method"\n>>print(Geekfgeeks.my_method.__qualname__)')
    # "My_Class.my_method"
    print(Geekfgeeks.my_method.__qualname__ + Fore.LIGHTGREEN_EX
          + '\n>>import inspect\n# initializing function\n>>def get_function_name():\n\t'
            '# get the frame object of the function\n\tframe = inspect.currentframe()\n\treturn frame.f_code.co_name\n'
            '# printing function name\n>>print("The name of function is : " + get_function_name()) # test_function')
    # printing function name
    print("The name of function is : " + get_function_name() + Fore.LIGHTGREEN_EX
          + '\n# This code is contributed by Edula Vinay Kumar Reddy\n' + Fore.CYAN + 'Descriptor.__name__\n'
          + Fore.LIGHTGREEN_EX
          + '# Python program showing\n# how to invoke descriptor\n>>def __getattribute__(self, key):\n\t'
            'v = object.__getattribute__(self, key)\n\tif hasattr(v, \'__get__\'):\n\t\t'
            'return v.__get__(None, self)\n\treturn v\n>>h = __getattribute__(vehicle, \'Car\')\n'
            '>>print(f\'The object vehicle has the attribute : {h}\\nThe name of the class is : {h.__name__}\')'
          )  # test_function
    h = __getattribute__(vehicle, 'Car')
    print(f'The object vehicle has the attribute : {h}\nThe name of the class is : {h.__name__}\n' + Fore.LIGHTGREEN_EX
          + '>>class Descriptor(object):\n\tdef __init__(self, name=\'\'):\n\t\tself.name = name\n\n\t' 
            'def __get__(self, obj, objtype):\n\t\treturn "{}for{}".format(self.name, self.name)\n\n\t'
            'def __set__(self, obj, name):\n\t\tif isinstance(name, str):\n\t\t\tself.name = name\n\t\telse:\n\t\t\t'
            'raise TypeError("Name should be string")\n>>class GFG(object):\n\tname = Descriptor()\n>>g = GFG()\n'
            '>>g.__name__ = "Geeks"\n>>print(g.__name__)')
    g = GFG()
    g.__name__ = "Geeks"
    print(f'{g.__name__}\n' + Fore.LIGHTGREEN_EX + '>>g = GFG()\ng.__name__ = "Computer"\n>>print(g.__name__)')
    g = GFG()
    g.__name__ = "Computer"
    print(f'{g.__name__}\n' + Fore.LIGHTGREEN_EX
          + '# Python program to explain property() function\n# Alphabet class\n>>class Alphabet:\n\t'
            'def __init__(self, value):\n\tself._value = value\n\n\t# getting the values\n\tdef getValue(self):\n\t\t'
            'print(\'Getting value\')\n\t\treturn self._value\n\n\t# setting the values\n\t'
            'def setValue(self, value):\n\t\tprint(\'Setting value to \' + value)\n\t\tself._value = value\n\n\t'
            '# deleting the values\n\tdef delValue(self):\n\t\tprint(\'Deleting value\')\n\t\tdel self._value\n\n\t'
            '__name__ = property(getValue, setValue, delValue, )\n# passing the value\n'
            '>>x = Alphabet(\'GeeksforGeeks\')')
    # passing the value
    x = Alphabet('GeeksforGeeks')
    print(Fore.LIGHTGREEN_EX + '>>print(x.__name__)\n' + Fore.WHITE + x.__name__ + Fore.LIGHTGREEN_EX
          + '\n>>x.__name__ = \'GfG\'')
    x.__name__ = 'GfG'
    print(Fore.LIGHTGREEN_EX + '>>del x.__name__')
    del x.__name__
    print(Fore.LIGHTGREEN_EX
          + '>>class AtAlphabet:\n\tdef __init__(self, value):\n\t\tself._value = value\n\n\t# getting the values\n\t'
            '@property\n\tdef __name__(self):\n\t\tprint(\'Getting value\')\n\t\treturn self._value\n\n\t'
            '# setting the values\n\t@__name__.setter\n\tdef __name__(self, value):\n\t\t'
            'print(\'Setting value to \' + value)\n\t\tself._value = value\n\n\t# deleting the values\n\t'
            '@value.deleter\n\tdef __name__(self):\n\t\tprint(\'Deleting value\')\n\t\tdel self._value\n'
            '# passing the value\n>>x = AtAlphabet(\'Peter\')')
    # passing the value
    x = AtAlphabet('Peter')
    print(Fore.LIGHTGREEN_EX + '>>print(x.value)\n' + Fore.WHITE + x.__name__ + Fore.LIGHTGREEN_EX
          + '\n>>x.__name__ = \'Diesel\'')
    x.__name__ = 'Diesel'
    print(Fore.LIGHTGREEN_EX + '>>del x.__name__')
    del x.__name__
    print(Fore.CYAN + 'Generator_Instance.__name__\n' + Fore.LIGHTGREEN_EX
          + '# A generator function that yields 1 for first time,\n# 2 second time and 3 third time\n'
            '>>def simpleGeneratorFun():\n\tyield 1\n\tyield 2\n\tyield 3\n'
            '>>spanish_numbers = [\'Uno\', \'Dos\', \'Tres\']\n>>gen = simpleGeneratorFun()\n'
            '# Driver code to check above generator function\n>>for value in simpleGeneratorFun():\n\t'
            'gen.__name__ = spanish_numbers[value-1]\n\tprint(gen.__name__, value)')
    spanish_numbers = ['Uno', 'Dos', 'Tres']
    gen = simpleGeneratorFun()
    # Driver code to check above generator function
    for value in simpleGeneratorFun():
        gen.__name__ = spanish_numbers[value-1]
        print(gen.__name__, value)
    print(Fore.LIGHTGREEN_EX
          + '# A Python program to demonstrate use of\n# generator object with next()\n# A generator function\n'
            '>>def simpleGeneratorFun():\n\tyield 1\n\tyield 2\n\tyield 3\n# x is a generator object\n'
            '>>x = simpleGeneratorFun()\n>>print(x.__name__)')
    # x is a generator object
    x = simpleGeneratorFun()
    print(x.__name__ + Fore.LIGHTGREEN_EX
          + '\n# Iterating over the generator object using next\n# In Python 3, __next__()'
            '\n>>for i in range(3):\n\tnumber = next(x)\n\tx.__name__ = spanish_numbers[number-1]\n\t'
            'print(x.__name__, number)')
    # Iterating over the generator object using next
    # In Python 3, __next__()
    for i in range(3):
        number = next(x)
        x.__name__ = spanish_numbers[number-1]
        print(x.__name__, number)
    print(Fore.LIGHTGREEN_EX
          + '>>def fib(limit):\n\ta, b = 0, 1\n\twhile b < limit:\n\t\tyield b\n\t\t'
            'a, b = b, a + b\n# Create a generator object\n>>x = fib(200)\n>>print(x.__name__)')
    # Create a generator object
    x = fib(200)
    print(x.__name__ + Fore.LIGHTGREEN_EX
          + '\n>>math_names = [\'Abacus\', \'Jacobian\', \'Arc\', \'Maximus\', \'Axiom\', \'Radius\', \'Calculus\','
            '\n\t\t\'Reckon\', \'Cipher\', \'Russell\', \'Converse\', \'Zahlen\']\n>>math_iter = iter(math_names)\n'
            '# Iterate over the generator object and print each value\n>>for i in x:\n\t'
            'x.__name__ = next(math_iter)\n\tprint(x.__name__.ljust(10), str(i).rjust(5))')
    math_names = ['Abacus', 'Jacobian', 'Arc', 'Maximus', 'Axiom', 'Radius', 'Calculus', 'Reckon', 'Cipher', 'Russell',
                  'Converse', 'Zahlen']
    math_iter = iter(math_names)
    for i in x:
        x.__name__ = next(math_iter)
        print(x.__name__.ljust(10), str(i).rjust(5))
    print(Fore.LIGHTGREEN_EX
          + '# generator expression\n>>generator_exp = (i * 5 for i in range(5) if i%2==0)\n'
            '>>print(generator_exp.__name__)')
    # generator expression
    generator_exp = (i * 5 for i in range(5) if i % 2 == 0)
    print(generator_exp.__name__ + Fore.LIGHTGREEN_EX
          + '\n>>girl_math = [\'Ajout\', \'Lemma\', \'Adele\']\n>>girl_iter = iter(girl_math)\n'
            '>>for i in generator_exp:\n\tgenerator_exp.__name__ = next(girl_iter)\n\t'
            'print(generator_exp.__name__.ljust(10), str(i).rjust(5))')
    girl_math = ['Ajout', 'Lemma', 'Adele']
    girl_iter = iter(girl_math)
    for i in generator_exp:
        generator_exp.__name__ = next(girl_iter)
        print(generator_exp.__name__.ljust(10), str(i).rjust(5))


def nest_func():
    def intern_func():
        pass
    return intern_func


def my_generator(n):
    # initialize counter
    value = 0
    # loop until counter is less than n
    while value < n:
        # produce the current value of the counter
        yield value
        # increment the counter
        value += 1


def PowTwoGen(max=0):
    n = 0
    while n < max:
        yield 2 ** n
        n += 1


def all_even():
    n = 0
    while n < 40:
        yield n
        n += 2


def fibonacci_numbers(nums):
    x, y = 0, 1
    for _ in range(nums):
        x, y = y, x+y
        yield x


def square(nums):
    for num in nums:
        yield num**2


def describe_definition_p__qualname__():
    print(Fore.YELLOW
          + 'definition.__qualname__ Test\n\thttps://peps.python.org/pep-3155/\n\t'
            'https://docs.python.org/3/howto/descriptor.html\n\thttps://www.programiz.com/python-programming/generator\n'
          + Fore.CYAN + 'Class.__qualname__\n' + Fore.LIGHTGREEN_EX
          + '>>class C:\n\tdef f():\n\t\tpass\n\tclass D:\n\t\tdef g():\n\t\t\tpass\n>>C.__qualname__\n'
          + Fore.WHITE + C.__qualname__ + Fore.LIGHTGREEN_EX + '\n>>C.f.__qualname__\n' + Fore.WHITE
          + C.f.__qualname__ + Fore.LIGHTGREEN_EX + '\n>>C.D.__name__\n' + Fore.WHITE + C.D.__qualname__
          + Fore.LIGHTGREEN_EX + '\n>>C.D.g.__qualname__\n' + Fore.WHITE + C.D.g.__qualname__ + Fore.CYAN
          + '\nFunction.__qualname__\n' + Fore.LIGHTGREEN_EX
          + '>>def nest_func():\n\tdef intern_func():\n\t\tpass\n\treturn intern_func\n>>nest_func.__qualname__\n'
          + Fore.WHITE + nest_func.__qualname__ + Fore.LIGHTGREEN_EX + '\n>>nest_func().__qualname__\n' + Fore.WHITE
          + nest_func().__qualname__ + Fore.LIGHTGREEN_EX
          + '\n>>class ValueDescriptor:\n\tlogging.basicConfig(level=logging.INFO)\n\n\t'
            'def __set_name__(self, owner, name):\n\t\tself.__qualname__ = f\'{owner.__qualname__}.{name}\'\n\t\t'
            'self.private_name = f\'_{owner.__qualname__}.{name}\'\n\n\tdef __get__(self, obj, objtype=None):\n\t\t'
            'if obj is None:\n\t\t\treturn self\n\t\tvalue = getattr(obj, self.private_name)\n\t\t'
            'logging.info(\'Accessing %r giving %r\', self.__qualname__, value)\n\t\treturn value\n\n\t'
            'def __set__(self, obj, value):\n\t\tlogging.info(\'Updating %r to %r\', self.__qualname__, value)\n\t\t'
            'setattr(obj, self.private_name, value)\n>>class MyValueClass:\n\tx = ValueDescriptor()\n\n\t'
            'def __init__(self, x):\n\t\tself.x = x\n>>obj = MyValueClass(10)')
    obj = MyValueClass(10)
    print(Fore.LIGHTGREEN_EX + '>>print(obj.x) # Accessing the descriptor')
    print(obj.x)  # Accessing the descriptor
    print(Fore.LIGHTGREEN_EX + '>>obj.x = 20 # Setting the descriptor')
    obj.x = 20  # Setting the descriptor
    print(Fore.LIGHTGREEN_EX + '>>print(obj.x)')
    print(obj.x)
    print(Fore.LIGHTGREEN_EX + '>>print(MyClass.x.__qualname__) # Accessing the qualname of the descriptor')
    print(f'{MyValueClass.x.__qualname__}')  # Accessing the qualname of the descriptor
    print(Fore.CYAN + 'Generator.__qualname__\n' + Fore.LIGHTGREEN_EX
          + 'def my_generator(n):\n\t# initialize counter\n\tvalue = 0\n\t# loop until counter is less than n\n\t'
            'while value < n:\n\t\t# produce the current value of the counter\n\t\tyield value\n\t\t'
            '# increment the counter\n\t\tvalue += 1\n# iterate over the generator object produced by my generator\n'
            '>>print(my_generator.__qualname__)\n' + Fore.WHITE + f'{my_generator.__qualname__}\n' + Fore.LIGHTGREEN_EX
          + '>># iterate over the generator object produced by my_generator\nfor value in my_generator(3):\n\t'
            '# print each value produced by generator\n\tprint(value)')
    # iterate over the generator object produced by my_generator
    for value in my_generator(3):
        # print each value produced by generator
        print(value)
    print(Fore.LIGHTGREEN_EX + '>>generator = my_range(3)\n>>print(generator.__qualname__)')
    generator = my_generator(3)
    print(generator.__qualname__ + Fore.LIGHTGREEN_EX + '\n>>print(next(generator))  # 0\n' + Fore.WHITE
          + f'{next(generator)}\n' + Fore.LIGHTGREEN_EX + '>>print(next(generator))  # 1\n' + Fore.WHITE
          + f'{next(generator)}\n' + Fore.LIGHTGREEN_EX + '>>print(next(generator))  # 2\n' + Fore.WHITE
          + f'{next(generator)}\n' + Fore.LIGHTGREEN_EX
          + '# create the generator object\n>>squares_generator = (i * i for i in range(5))\n'
            '>>squares_generator.__qualname__')
    # create the generator object
    squares_generator = (i * i for i in range(5))
    print(squares_generator.__qualname__)
    print(Fore.LIGHTGREEN_EX
          + '# iterate over the generator and print the values\n>>for i in squares_generator:\n\tprint(i)')
    # iterate over the generator and print the values
    for i in squares_generator:
        print(i)
    print(Fore.LIGHTGREEN_EX
          + '>>class PowTwo:\n\tdef __init__(self, max=0):\n\t\tself.n = 0\n\t\tself.max = max\n\t\t'
            'self.__qualname__ = type(self).__qualname__\n\n\tdef __iter__(self):'
            '\n\t\treturn self\n\n\tdef __next__(self):\n\t\tif self.n > self.max:\n\t\t\traise StopeIteration\n\t\t'
            'result = 2 ** self.n\n\t\tself.n += 1\n\t\treturn result\n>>the_power = PowTwo(12)\n'
            '>>print(the_power.__qualname__)')
    the_power = PowTwo(12)
    print(the_power.__qualname__ + Fore.LIGHTGREEN_EX
          + '\n>>the_power_iter = iter(the_power)\n>>try:\n\twhile True:\n\t\tprint(next(the_power_iter))\n'
            'except StopIteration as e:\n\tpass')
    the_power_iter = iter(the_power)
    try:
        while True:
            print(next(the_power_iter))
    except StopIteration as e:
        pass
    print(Fore.LIGHTGREEN_EX
          + '>>def PowTwoGen(max=0):\n\tn = 0\n\twhile n < max:\n\t\tyield 2 ** n\n\t\tn += 1\n'
            '>>record = PowTwoGen(14)\n>>print(record.__qualname__)')
    record = PowTwoGen(14)
    print(record.__qualname__ + Fore.LIGHTGREEN_EX + '\n>>for n in record:\n\tprint(n)')
    for n in record:
        print(n)
    print(Fore.LIGHTGREEN_EX
          + '>>def all_even():\n\tn = 0\n\twhile True:  # not the actual code n < 40 is the actual code.\n\t\t'
            'yield n\n\t\tn += 2\n>>infinite = all_even()\n>>print(infinite.__qualname__)')
    infinite = all_even()
    print(infinite.__qualname__ + Fore.LIGHTGREEN_EX + '\n>>for i in infinite:\n\tprint(i)')
    for i in infinite:
        print(i)
    print(Fore.LIGHTGREEN_EX
          + '>>def fibonacci_numbers(nums):\n\tx, y = 0, 1\n\tfor _ in range(nums):\n\t\tx, y = y, x+y\n\t\tyield x\n'
            '>>def square(nums):\n\tfor num in nums:\n\t\tyield num**2\n>>before_line = fibonacci_numbers(10)\n'
            '>>pipeline = square(before_line)\n>>print(before_line.__qualname__, pipeline.__qualname__)')
    before_line = fibonacci_numbers(10)
    pipeline = square(before_line)
    print(before_line.__qualname__, pipeline.__qualname__ + Fore.LIGHTGREEN_EX
          + '\n>>print(sum(pipeline))  # Output: 4895\n' + Fore.WHITE + '{0}'.format(sum(pipeline)))  # Output: 4895


def describe_del_d_key():
    print(Fore.YELLOW
          + 'del d[key] Test\n\tFrom https://www.geeksforgeeks.org/python-ways-to-remove-a-key-from-dictionary/\n'
          + Fore.LIGHTGREEN_EX
          + '# Initializing dictionary\n>>test_dict = {"Arushi": 22, "Mani": 21, "Haritha": 21}\n'
            '# Printing dictionary before removal\n'
            '>>print("The dictionary before performing remove is : ", test_dict)')
    # Initializing dictionary
    test_dict = {"Arushi": 22, "Mani": 21, "Haritha": 21}
    # Printing dictionary before removal
    print(f'The dictionary before performing remove is : {test_dict}\n' + Fore.LIGHTGREEN_EX
          + '# Using del to remove a dict\n# removes Mani\n>>del test_dict[\'Mani\']\n'
            '# Printing dictionary after removal\n>>print("The dictionary after remove is : ", test_dict)')
    # Using del to remove a dict
    # removes Mani
    del test_dict['Mani']
    # Printing dictionary after removal
    print(f'The dictionary after remove is : {test_dict}\n' + Fore.LIGHTGREEN_EX
          + '# Using del to remove a dict\n# raises exception\n>>try:\n\tdel test_dict[\'Mani\']\n'
            'except KeyError as e:\n\tprint(traceback.format_exc())')
    # Using del to remove a dict
    # raises exception
    try:
        del test_dict['Mani']
    except KeyError as e:
        print(traceback.format_exc())


def describe_del_start_end_step():
    print(Fore.YELLOW + 'del s[i:j:k] Test\n' + Fore.LIGHTGREEN_EX
          + '>>german_name = [\'Carl\', \'Felix\', \'Albert\', \'Christoph\', \'Otto\', \'Albrecht\', \'Alexander\', \n'
            '\'Frederick\', \'Hans\', \'Henry\', \'Klaus\', \'Leo\', \'Louis\', \'Adalbert\', \'Adolf\', \'Alwin\', \n'
            '\'Andreas\', \'Anselm\', \'Anton\', \'Armin\', \'Axel\', \'Conrad\', \'Adam\']\n'
            '>>print("The original list is : \\n", \'\\n\'.join(textwrap.wrap(str(german_name), 80)))')
    german_name = ['Carl', 'Felix', 'Albert', 'Christoph', 'Otto', 'Albrecht', 'Alexander', 'Frederick', 'Hans',
                   'Henry', 'Klaus', 'Leo', 'Louis', 'Adalbert', 'Adolf', 'Alwin', 'Andreas', 'Anselm', 'Anton',
                   'Armin', 'Axel', 'Conrad', 'Adam']
    print("The original list is : \n", '\n'.join(textwrap.wrap(str(german_name), 80)) + Fore.LIGHTGREEN_EX
          + '\n>>del german_name[5:15:3]\n'
            '>>print("The cropped list is : ", \'\\n\'.join(textwrap.wrap(str(german_name)), 80))')
    del german_name[5:15:3]
    print("The cropped list is : \n", '\n'.join(textwrap.wrap(str(german_name), 80)))


def describe_del_start_end():
    print(Fore.YELLOW + 'del s[i:j] Test\n' + Fore.LIGHTGREEN_EX
          + '>>chinese_name = [\'Chanchan\', \'Chanxin\', \'Diedie\', \'Diwei\', \'Jia Hao\', \'Jia Wei\',\n'
            '\'Jun De\', \'Ming Tao\', \'Yuechan\', \'Ah Lam\', \'An\', \'Bai\', \'Bao\', \'Bo\', \'Caihong\',\n'
            '\'Chang\', \'Chow\', \'Chunhua\', \'Ehuang\']\n'
            '>>print("The original list is : \\n", \'\\n\'.join(textwrap.wrap(str(chinese_name), 80)))')
    chinese_name = ['Chanchan', 'Chanxin', 'Diedie', 'Diwei', 'Jia Hao', 'Jia Wei', 'Jun De', 'Ming Tao', 'Yuechan',
                    'Ah Lam', 'An', 'Bai', 'Bao', 'Bo', 'Caihong', 'Chang', 'Chow', 'Chunhua', 'Ehuang']
    print("The original list is : \n", '\n'.join(textwrap.wrap(str(chinese_name), 80)) + Fore.LIGHTGREEN_EX
          + '\n>>del chinese_name[4:8]\n'
            '>>print("The cropped list is : \\n", \'\\n\'.join(textwrap.wrap(str(chinese_name), 80)))')
    del chinese_name[4:8]
    print("The cropped list is : \n", '\n'.join(textwrap.wrap(str(chinese_name), 80)) + Fore.LIGHTGREEN_EX
          + '\n>>chinese_name[4:8] = []\n'
            '>>print("The second cropped list is : \\n", \'\\n\'.join(textwrap.wrap(str(chinese_name), 80)))')
    chinese_name[4:8] = []
    print("The second cropped list is : \n", '\n'.join(textwrap.wrap(str(chinese_name), 80)))


def describe_delattr():
    print(Fore.YELLOW
          + 'delattr(object, name) Method Test\n\t'
            'From https://www.programiz.com/python-programming/methods/built-in/delattr\n' + Fore.LIGHTGREEN_EX
          + 'class Coordinate:\n\tx = 10\n\ty = -5\n\tz = 0\n>>point1 = Coordinate()\n>>print(\'x = \', point.x)')
    point1 = Coordinate()
    print(f'x = {point1.x}\n' + Fore.LIGHTGREEN_EX + '>>print(\'y = \', point1.y)\n' + Fore.WHITE + f'y = {point1.y}\n'
          + Fore.LIGHTGREEN_EX + '>>print(\'z = \', point1.z)\n' + Fore.WHITE + f'z = {point1.z}\n' + Fore.LIGHTGREEN_EX
          + '>>delattr(Coordinate, \'z\')\n>>print(\'--After deleting z attribute--\')')
    delattr(Coordinate, 'z')
    print('--After deleting z attribute--\n' + Fore.LIGHTGREEN_EX + '>>print(\'x = \', point1.x)\n' + Fore.WHITE
          + f'x = {point1.x}\n' + Fore.LIGHTGREEN_EX + '>>print(\'y = \', point1.y)\n' + Fore.WHITE
          + f'y = {point1.y}\n' + Fore.LIGHTGREEN_EX
          + '# Raises Error\n>>try:\n\tprint(\'z = \', point1.z)\nexcept AttributeError as e:\n\t'
            'print(traceback.format_exc()')
    # Raises Error
    try:
        print('z = ', point1.z)
    except AttributeError as e:
        print(traceback.format_exc())
    print(Fore.LIGHTGREEN_EX
          + '>>point1 = Coordinate()\n# Redefine the attribute after deleting it\n>>Coordinate.z\n'
            '>>print(\'x = \', point1.x)')
    point1 = Coordinate()
    # Redefine the attribute after deleting it
    Coordinate.z = 0
    print(f'x = {point1.x}\n' + Fore.LIGHTGREEN_EX + '>>print(\'y = \', point1.y)\n' + Fore.WHITE
          + f'y = {point1.y}\n' + Fore.LIGHTGREEN_EX + '>>print(\'z = \', point1.z)\n' + Fore.WHITE
          + f'z = {point1.z}\n' + Fore.LIGHTGREEN_EX
          + '# Deleting attribute z\n>>del Coordinate.z\n>>print(\'--After deleting z attribute--\')')
    del Coordinate.z
    print('--After deleting z attribute--\n' + Fore.LIGHTGREEN_EX + '>>print(\'x = \', point1.x)\n' + Fore.WHITE
          + f'x = {point1.x}\n' + Fore.LIGHTGREEN_EX + '>>print(\'y = \', point1.y)\n' + Fore.WHITE
          + f'y = {point1.y}\n' + Fore.LIGHTGREEN_EX
          + '# Raises Attribute Error\n>>try:\n\t>>print(\'z = \', point1.z)\nexcept AttributeError as e:\n\t'
            'print(traceback.format_exc())')
    # Raises Attribute Error
    try:
        print('z = ', point1.z)
    except AttributeError as e:
        print(traceback.format_exc())


def describe_dict_clear():
    print(Fore.YELLOW
          + 'Dictionary Version clear() Method Test\n\tFrom https://www.geeksforgeeks.org/python-dictionary-clear/\n'
          + Fore.LIGHTGREEN_EX
          + '# Python program to demonstrate working of\n# dictionary clear()\n>>text = {1: "geeks", 2: "for"}\n'
            '>>text.clear()\n>>print(\'text =\', text)')
    # Python program to demonstrate working of
    # dictionary clear()
    text = {1: "geeks", 2: "for"}
    text.clear()
    print('text =', text, Fore.LIGHTGREEN_EX
          + '\n# Python code to demonstrate difference\n# clear and {}.\n>>text1 = {1: "geeks", 2: "for"}\n'
            '>>text2 = text1\n# Using clear makes both text1 and text2\n# empty.\n>>text1.clear()\n'
            '>>print(\'After removing items using clear()\')')
    # Python code to demonstrate difference
    # clear and {}.
    text1 = {1: "geeks", 2: "for"}
    text2 = text1
    # Using clear makes both text1 and text2
    # empty.
    text1.clear()
    print('After removing items using clear()\n' + Fore.LIGHTGREEN_EX + '>>print(\'text1 =\', text1)\n' + Fore.WHITE
          + f'text1 = {text1}\n' + Fore.LIGHTGREEN_EX + '>>print(\'text2 =\', text2)\n' + Fore.WHITE
          + f'text2 = {text2}\n' + Fore.LIGHTGREEN_EX
          + '>>text1 = {1: "one", 2: "two"}\n>>text2 = text1\n# This makes only text1 empty.\n>>text1 = {}\n'
            '>>print(\'After removing items by assigning {}\')')
    text1 = {1: "one", 2: "two"}
    text2 = text1
    # This makes only text1 empty.
    text1 = {}
    print('After removing items by assigning {}\n' + Fore.LIGHTGREEN_EX + '>>print(\'text1 =\', text1)\n' + Fore.WHITE
          + f'text1 = {text1}\n' + Fore.LIGHTGREEN_EX + '>>print(\'text2 =\', text2)\n' + Fore.WHITE
          + f'text2 = {text2}')


def describe_dict_copy():
    print(Fore.YELLOW
          + 'Dictionary Version copy() Method Test\n\t'
            'From https://www.programiz.com/python-programming/methods/dictionary/copy\n' + Fore.LIGHTGREEN_EX
          + '>>original_marks = {\'Physics\': 67, \'Maths\': 87}\n>>copied_marks = original_marks.copy()\n'
            '>>print(\'Original Marks:\', original_marks)')
    original_marks = {'Physics': 67, 'Maths': 87}
    copied_marks = original_marks.copy()
    print('Original Marks:', original_marks, Fore.LIGHTGREEN_EX + '\n>>print(\'Copied Marks:\', copied_marks)',
          '\nCopied Marks:', copied_marks, Fore.LIGHTGREEN_EX
          + '\n>>original = {1: \'one\', 2: \'two\'}\n>>new = original.copy()\n>>print(\'Original: \', original)')
    original = {1: 'one', 2: 'two'}
    new = original.copy()
    print('Original: ', original, Fore.LIGHTGREEN_EX + '\n>>print(\'New: \', new)', '\nNew: ', new, Fore.LIGHTGREEN_EX
          + '\n# removing all elements from the list\n>>new.clear()\nprint(\'new: \', new)')
    new.clear()
    print('new: ', new, Fore.LIGHTGREEN_EX + '\n>>print(\'original: \', original)', '\noriginal: ', original,
          Fore.LIGHTGREEN_EX
          + '\n>>new = original\n# removing all elements from the list\n>>new.clear()\n>>print(\'new: \', new)')
    new = original
    # removing all elements from the list
    new.clear()
    print('new: ', new, Fore.LIGHTGREEN_EX + '\n>>print(\'original: \', original)', '\noriginal: ', original)


def describe_dictview_mapping():
    print(Fore.YELLOW + 'dictview.mapping Test', Fore.LIGHTGREEN_EX
          + '\n>>dishes = {\'Plates\': 4, \'Kettle\': 1, \'Teaspoon\': 2, \'Cutlery\': 12, \'Glassware\': 5}\n'
            '>>keys = dishes.keys()\n>>values = dishes.values()\n>>items = dict.items()\n'
            '>>print(\'keys: \', keys, \'\\nvalues:\', values, \'\\nitems: \', items, \n'
            '  Fore.CYAN + \'\\nMapping Proxy\\\'s\', \'keys: \', keys.mapping, \'\\nvalues:\', values.mapping,\n'
            '  \'\\nitems: \', items.mapping)')
    dishes = {'Plates': 4, 'Kettle': 1, 'Teaspoon': 2, 'Cutlery': 12, 'Glassware': 5}
    keys = dishes.keys()
    values = dishes.values()
    items = dishes.items()
    print('keys: ', keys, '\nvalues: ', values, '\nitems: ', items, Fore.CYAN + '\nMapping Proxy\'s',
          '\nkeys: ', keys.mapping, '\nvalues:', values.mapping, '\nitems: ', items.mapping, Fore.LIGHTGREEN_EX
          + '\n>>print(keys.mapping[\'Cutlery\'])', f"\n{keys.mapping['Cutlery']}", Fore.LIGHTGREEN_EX
          + '\n>>print(values.mapping[\'Glassware\'])', f"\n{values.mapping['Glassware']}", Fore.LIGHTGREEN_EX
          + '\n>>print(items.mapping[\'Kettle\']', f"\n{items.mapping['Kettle']}", Fore.LIGHTGREEN_EX
          + '>>map = keys.mapping\n>>try:\n\tmap[\'Kettle\'] = 4\nexcept AttributeError as e:\n\t'
            'print(traceback.format_exc())')
    map = keys.mapping
    try:
        map['Kettle'] = 4
    except TypeError as e:
        print(traceback.format_exc())


def describe_difference():
    print(Fore.YELLOW
          + 'difference(*others) Method Test\n\tFrom https://www.w3schools.com/python/ref_set_difference.asp\n'
          + Fore.LIGHTGREEN_EX
          + '>>x = {"apple", "banana", "cherry"}\n>>y = {"google", "microsoft", "apple"}\n>>z = x.difference(y)\n'
            '>>print(z)')
    x = {"apple", "banana", "cherry"}
    y = {"google", "microsoft", "apple"}
    z = x.difference(y)
    print(z, Fore.LIGHTGREEN_EX
          + '\n>>a = {"apple", "banana", "cherry"}\n>>b = {"google", "microsoft", "apple"}\n>>myset = a - b\n'
            '>>print(myset)')
    a = {"apple", "banana", "cherry"}
    b = {"google", "microsoft", "apple"}
    myset = a - b
    print(myset, Fore.LIGHTGREEN_EX
          + '\n>>c = {"cherry", "micra", "bluebird"}\n>>myset = a.difference(b, c)\n>>print(myset)')
    c = {"cherry", "micra", "bluebird"}
    myset = a.difference(b, c)
    print(myset, Fore.LIGHTGREEN_EX + '\n>>myset = a - b - c\n>>print(myset)')
    myset = a - b - c
    print(myset, Fore.LIGHTGREEN_EX + '\n>>z = y.difference(x)\n>>print(z)')
    z = y.difference(x)
    print(z)


def describe_difference_update():
    print(Fore.YELLOW
          + 'difference_update(*others) Method Test\n\t'
            'From https://www.geeksforgeeks.org/python-set-difference_update/\n\t'
            'https://www.w3schools.com/python/ref_set_difference_update.asp\n' + Fore.LIGHTGREEN_EX
          + '# Python code to get the difference between two sets\n'
            '# using difference_update() between set A and set B\n# Driver Code\n>>A = {10, 20, 30, 40, 80}\n'
            '>>B = {100, 30, 80, 40, 60}\n# Modifies A and returns None\n>>A.difference_update(B)\n'
            '# Prints the modified set\n>>print(A)')
    # Python code to get the difference between two sets
    # using difference_update() between set A and set B
    # Driver Code
    A = {10, 20, 30, 40, 80}
    B = {100, 30, 80, 40, 60}
    # Modifies A and returns None
    A.difference_update(B)
    # Prints the modified set
    print(A, Fore.LIGHTGREEN_EX
          + '\n>>all = {"apple", "banana", "cherry"}\n>>default = all.copy()\n'
            '>>bad = {"google", "microsoft", "apple"}\n>>all -= bad\n>>print(all)')
    all = {"apple", "banana", "cherry"}
    default = all.copy()
    bad = {"google", "microsoft", "apple"}
    all -= bad
    print(all, Fore.LIGHTGREEN_EX
          + '\n>>all = default.copy()\n>>cake = {"cherry", "micra", "bluebird"}\n'
            '>>all.difference_update(b, c)\n>>print(all)')
    all = default.copy()
    cake = {"cherry", "micra", "bluebird"}
    all.difference_update(bad, cake)
    print(all, Fore.LIGHTGREEN_EX + '\n>>all = default.copy()\n>>all -= bad | cake\n>>print(all)')
    all = default.copy()
    all -= bad | cake
    print(all)


def if_not_none(item):
    if item is not None:
        print(item)


def describe_dir():
    print(Fore.YELLOW
          + 'dir(), dir(object) Method Test\nFrom https://www.geeksforgeeks.org/python-dir-function/\n\t'
            'https://medium.com/@satishgoda/python-attribute-access-using-getattr-and-getattribute-6401f7425ce6'
          + Fore.LIGHTGREEN_EX
          + '# Python3 code to demonstrate dir()\n# when no parameters are passed\n'
            '# Note that we have not imported any modules\n>>print(dir())')
    # Python3 code to demonstrate dir()
    # when no parameters are passed
    # Note that we have not imported any modules
    print(dir(), Fore.LIGHTGREEN_EX + '# Now let\'s import two modules\n>>import random\n>>import math\n>>print(dir())')
    # Now let's import two modules
    import random
    import math
    print(dir(), Fore.LIGHTGREEN_EX
          + '# Python3 code to demonstrate dir() function\n# when a module Object is passed as parameter.\n'
            '# import the random module\n>>import random\n# Prints list which contains names of\n'
            '# attributes in random function\n>>print("The content of the random library are::")')
    # Python3 code to demonstrate dir() function
    # when a module Object is passed as parameter.
    # I have already imported the random module.
    # Prints list which contains names of
    # attributes in random function
    print("The contents of the random library are::\n" + Fore.LIGHTGREEN_EX
          + '# module Object is passed as parameter\n>>print(dir(random))')
    # module Object is passed as parameter
    print(dir(random), Fore.LIGHTGREEN_EX
          + '\n# When a list object is passed as\n# parameters for the dir() function\n# A list, which contains\n'
            '# a few random values\n>>geeks = ["geeksforgeeks", "gfg", "Computer Science", "Data Structures",'
            ' "Algorithms"]\n# dir() will also list out common\n# attributes of the dictionary\n'
            '>>d = {}  # empty dictionary\n# dir() will return all the available\n# list methods in current local scope'
            '\n>>print(dir(geeks))')
    # When a list object is passed as
    # parameters for the dir() function
    # A list, which contains
    # a few random values
    geeks = ["geeksforgeeks", "gfg", "Computer Science", "Data Structures", "Algorithms"]
    # dir() will also list out common
    # attributes of the dictionary
    d = {}  # empty dictionary
    # dir() will return all the available
    # list methods in current local scope
    print(dir(geeks), Fore.LIGHTGREEN_EX + '\n>>print(dir(d))', f'\n{dir(d)}', Fore.LIGHTGREEN_EX
          + '\n# Creation of a simple class with __dir__\n# method to demonstrate it\'s working\n'
            '>>class Supermarket:\n\t# Function __dir__ which list all\n\t# the base attributes to be used.\n\t'
            'def __dir__(self):\n\t\treturn [\'customer_name\', \'product\',\n\t\t\t'
            '\'quantity\', \'price\', \'date\']\n\n# user-defined object of class supermarket\n'
            '>>my_cart = Supermarket()\n# listing out the dir() method\n>>print(dir(my_cart))')
    # user-defined object of class supermarket
    my_cart = Supermarket()
    # listing out the dir() method
    print(dir(my_cart), Fore.LIGHTGREEN_EX
          + '\n>>class Yeah(object):\n\tdef __init__(self, name):\n\t\tself.name = name\n\n\t'
            '# Gets called when an attribute is accessed\n\tdef __getattribute__(self, item):\n\t\t'
            'print(\'__getattribute__\', item)\n\t\t# Calling the super class to avoid recursion\n\t\t'
            'return super(Yeah, self).__getattribute__(item)\n\n\t'
            '# Gets called when the item is not found via __getattribute__\n\tdef __getattr__(self, item):\n\t\t'
            'print(\'__getattr__\', item)\n\t\treturn super(Yeah, self).__setattr__(item, \'orphan\')\n'
            '>>y1 = Yeah(\'yes\')\n>>y1.name')
    y1 = Yeah('yes')
    if_not_none(y1.name)
    print(Fore.LIGHTGREEN_EX + '\n>>y1.foo')
    if_not_none(y1.foo)
    print(Fore.LIGHTGREEN_EX + '\n>>y1.foo')
    if_not_none(y1.foo)
    print(Fore.LIGHTGREEN_EX + '\n>>y1.goo')
    if_not_none(y1.goo)
    print(Fore.LIGHTGREEN_EX + '\n>>y1.__dict__')
    if_not_none(y1.__dict__)
    print(Fore.LIGHTGREEN_EX + '\n>>dir(y1)')
    print(f'\n{dir(y1)}')


def describe_discard():
    print(Fore.YELLOW
          + 'discard(elem) Method Test\n\tFrom https://www.programiz.com/python-programming/methods/set/discard'
          + Fore.LIGHTGREEN_EX
          + '>>numbers = {2, 3, 4, 5}\n# discards 3 from the set\n>>numbers.discard(3)\n'
            '>>print(\'Set after discard:\', numbers)')
    numbers = {2, 3, 4, 5}
    # discards 3 from the set
    numbers.discard(3)
    print('Set after discard:', numbers, Fore.LIGHTGREEN_EX
          + '\n>>same_numbers = {2, 3, 5, 4}\n>>print(\'Set before discard:\', same_numbers)')
    same_numbers = {2, 3, 5, 4}
    print('Set before discard:', same_numbers, Fore.LIGHTGREEN_EX
          + '\n# discard the item that doesn\'t exist in set\n>>same_numbers.discard(10)\n'
            '>>print(\'Set after discard:\', same_numbers)')
    # discard the item that doesn't exist in set
    same_numbers.discard(10)
    print('Set after discard:', same_numbers)


def describe_divmod():
    print(Fore.YELLOW + 'divmod(a, b) Method Test\n\tFrom https://www.geeksforgeeks.org/divmod-python-application/\n'
          + Fore.LIGHTGREEN_EX
          + '# Python3 code to illustrate divmod()\n# divmod() with int\n>>print(\'(5, 4) = \', divmod(5, 4))')
    # Python3 code to illustrate divmod()
    # divmod() with int
    print('(5, 4) = ', divmod(5, 4), Fore.LIGHTGREEN_EX +
          '\n>>print(\'(10, 16) = \', divmod(10, 16))', '\n(10, 16) = ', divmod(10, 16), Fore.LIGHTGREEN_EX +
          '\n>>print(\'(11, 11) = \', divmod(11, 11))', '\n(11, 11) = ', divmod(11, 11), Fore.LIGHTGREEN_EX +
          '\n>>print(\'(15, 13) = \', divmod(15, 13))', '\n(15, 13) = ', divmod(15, 13), Fore.LIGHTGREEN_EX +
          '\n# divmod() with int and Floats\n>>print(\'(8.0, 3) = \', divmod(8.0, 3))')
    # divmod() with int and Floats
    print('(8.0, 3) = ', divmod(8.0, 3), Fore.LIGHTGREEN_EX + '\n>>print(\'(3, 8.0) = \', divmod(3, 8.0))',
          '\n(3, 8.0) = ', divmod(3, 8.0), Fore.LIGHTGREEN_EX + '\n>>print(\'(7.5, 2.5) = \', divmod(7.5, 2.5))',
          '\n(7.5, 2.5) = ', divmod(7.5, 2.5), Fore.LIGHTGREEN_EX + '\n>>print(\'(2.6, 10.7) = \', divmod(2.6, 0.5))',
          '\n(2.6, 10.7) = ', divmod(2.6, 0.5), Fore.LIGHTGREEN_EX +
          '\n# Python code to find if a number is\n# prime or not using divmod()\n# Given integer\n>>n = 15\n>>x = n\n'
          '# Initialising counter to 0\n>>count = 0\n>>while x != 0:\n\tp, q = divmod(n, x)\n\tx -= 1\n\t'
          'if q == 0:\n\t\tcount += 1\n>>if count > 2:\n\tprint(\'Not Prime\')\n  else:\nprint(\'Prime\')')
    # Python code to find if a number is
    # prime or not using divmod()
    # Given integer
    n = 15
    x = n
    # Initializing counter to 0
    count = 0
    while x != 0:
        p, q = divmod(n, x)
        x -= 1
        if q == 0:
            count += 1
    if count > 2:
        print('Not Prime')
    else:
        print('Prime')
    print(Fore.LIGHTGREEN_EX
          + '# Sum of digits of a number using divmod\n>>num = 86\n>>sums = 0\n>>while num != 0:\n\t'
            'use = divmod(num, 10)\n\tdig = use[1]\n\tsums = sums + dig\n\tnum = use[0]\nprint(sums)')
    # Sum of digits of a number using divmod
    num = 86
    sums = 0
    while num != 0:
        use = divmod(num, 10)
        dig = use[1]
        sums = sums + dig
        num = use[0]
    print(sums, Fore.LIGHTGREEN_EX +
          '\n# reversing a number using divmod\n>>num = 132\n>>pal = 0\n>>while num != 0:\n\tuse = divmod(num, 10)\n\t'
          'dig = use[1]\n\tpal = pal*10+dig\n\tnum = use[0]\n>>print(pal)')
    # reversing a number using divmod
    num = 132
    pal = 0
    while num != 0:
        use = divmod(num, 10)
        dig = use[1]
        pal = pal * 10 + dig
        num = use[0]
    print(pal, Fore.LIGHTGREEN_EX + '\n>>a = 15\n>>b = 6\n>>print(\'divmod(15, 6) = \', divmod(15, 6))')
    a = 15
    b = 6
    print('divmod(15, 6) =', divmod(15, 6), Fore.LIGHTGREEN_EX
          + '\n>>print(\'(a // b, a % b) =\', (a // b, a % b))',
          '\n(a // b, a % b) =', (a // b, a % b), Fore.LIGHTGREEN_EX + '\n>>print(\'(q, a % b) =\', (a // b, a % b))',
          '\n(q, a % b) =', (a // b, a % b), Fore.LIGHTGREEN_EX
          + '\n>>import math'
            '\n>>print(\'q is usually math.floor(a / b) = \', math.floor(a / b), \'or may be 1 less. q =\', a // b)',
          '\nq is usually math.floor(a / b)', math.floor(a / b), 'or may be 1 less. q =', a // b, Fore.LIGHTGREEN_EX
          + '\n>>print(\'q * b + a % b =\', (a // b) * b + a % b, \'= (close to a)\')', '\nq * b + a % b =',
          (a // b) * b + a % b, '= (close to a)', Fore.LIGHTGREEN_EX
          + '\n>>print(\'If a =\', a, \'is non-zero than b =\', b, \'has the same sign as a.\')', '\nIf a =', a,
          'is non-zero than b =', b, 'has the sme sign as a.', Fore.LIGHTGREEN_EX
          + '\n>>print(\'0 <= abs(a % b) < abs(b) is equivalent to\', 0 <= abs(a % b) < abs(b))',
          '\n0 <= abs(a % b) < abs(b) is equivalent to', 0 <= abs(a % b) < abs(b))


def describe_double_quotes():
    print(Fore.YELLOW + 'Double quotes', Fore.LIGHTGREEN_EX + '\n>>print("Single \'in\' the double")',
          "\nSingle 'in' the double")


def inject(get_next_item: Callable[..., str]) -> None:
    ...


def fool(x: ...) -> None:
    ...


# style1
def hen():
    pass


# style2
def van():
    ...


def hat(x = ...):
    return x


def describe_ellipsis():
    print(Fore.YELLOW
          + 'Ellipsis Test\n\tFrom https://www.geeksforgeeks.org/what-is-three-dots-or-ellipsis-in-python3/',
          Fore.LIGHTGREEN_EX
          + '\n>>> condition = True\n>>> if condition:\n...\t'
            'print("GeeksforGeeks")  # Ellipsis ("...") is used as a secondary prompt, primary prompt is >>>\n...')
    condition = True
    if condition:
        print("GeeksforGeeks")
    print(Fore.LIGHTGREEN_EX
          + '# importing numpy\n>>import numpy as np\n>>array_test = np.random.rand(2, 2, 2, 2)\n'
            '>>print(array_test[..., 0])')
    array_test = np.random.rand(2, 2, 2, 2)
    print(array_test[..., 0], Fore.LIGHTGREEN_EX + '\n>>print(array_test[Ellipsis, 0])',
          '\n{0}'.format(array_test[Ellipsis, 0]), Fore.LIGHTGREEN_EX
          + '\n>>from typing import Callable\n>>import inspect\n'
            '>>def inject(get_next_item: Callable[..., str]) -> None:\n\t...\n# Argument type is assumed as type: Any\n'
            '>>def foo(x: ...) -> None:\n\t...\n>>print(inspect.signature(inject), inspect.signature(fool))')
    print(inspect.signature(inject), inspect.signature(fool), Fore.LIGHTGREEN_EX
          + '\n>>class flow:\n\n\t# (using "value: Any" to allow arbitrary types)\n\t'
            'def __understand__(self, name: str, value: ...) -> None: ...\n>>friction = flow()\n'
            '>>print(inspect.signature(friction.__understand__))')
    friction = flow()
    print(inspect.signature(friction.__understand__), Fore.LIGHTGREEN_EX
          + '\n# style1\ndef hen():\n\tpass\n# style2\ndef van():\n\t...\n# both the styles are same\n'
            '>>print(inspect.signature(hen), inspect.signature(van))',
          '\n{0} {1}'.format(inspect.signature(hen), inspect.signature(van)), Fore.LIGHTGREEN_EX
          + '\n>>def hat(x = ...):\n\treturn x\n>>print(hat())', '\n'.format(hat()), Fore.LIGHTGREEN_EX
          + '\n>>l = np.random.randint(0, 50, 4)\n>>print(l)')
    l = np.random.randint(0, 50, 4)
    print(l, Fore.LIGHTGREEN_EX
          + '\n>>try:\n\tl[..., 1, ...]\nexcept IndexError as e:\n\tprint(traceback.format_exc())')
    try:
        print(l[..., 1, ...])
    except IndexError as e:
        print(traceback.format_exc())


def describe_emscription():
    print(Fore.YELLOW
          + 'Emscription Test\n\tFrom google, download emscription extensions here\n'
            'https://emscripten.org/docs/getting_started/downloads.html.\n\t'
            'Install wasmtime by running the command >>>pip install wasmtime\n\tin command prompt.\n'
          + Fore.CYAN + 'CPP file\n' + Fore.LIGHTGREEN_EX
          + '// c_function.cpp\n\nextern "C" {\n\tint add(int a, int b) {\n\t\treturn a + b;\n\t}\n}\n' + Fore.CYAN
          + 'Command Prompt\n' + Fore.LIGHTGREEN_EX
          + 'C:User\\..>emcc c_function.cpp -03 -c -o c_function.wasm --no_entry\n' + Fore.CYAN
          + 'Python' + Fore.LIGHTGREEN_EX
          + '>>import wasmtime\n# Load the WebAssembly module\n>>engine = wasmtime.Engine()\n'
            '>>store = wasmtime.Store(engine)\n>>module = wasmtime.Module.from_file(store, "c_function.wasm")\n'
            '# Create an instance of the module\n>>instance = wasmtime.Instance(store, module, [])\n'
            '# Call the exported function\n>>add_func = instance.exports["add"]\n>>result = add_func(2, 3)\n'
            '>>print(result)')
    # Load the WebAssembly module
    store = Store()
    module = Module.from_file(store.engine, 'c_function.wasm')
    # Create an instance of the module
    instance = Instance(store, module, )
    # Call the exported function
    add_func = instance.exports(store)['add']
    result = add_func(store, 2, 3)
    print(result)



def main():
    describe_bool()
    describe_bytearray()
    describe_bytes()
    describe_complex()
    describe_dict()
    describe_frozenset()
    describe_int_func()
    describe_list()
    describe_memory_view()
    describe_object()
    describe_property()
    describe_range()
    describe_set()
    describe_slice()
    describe_str()
    describe_super()
    describe_tuple()
    describe_type()
    describe_bases()
    describe_mro()
    describe_subclasses_class()
    describe_not_equal_opterator()
    describe_unchanged()
    describe_negated()
    describe_less_than()
    describe_less_or_equal()
    describe_equal()
    describe_greater_than()
    describe_greater_or_equal()
    describe_class_method()
    describe_static_method()
    describe_debug()
    describe_exporter()
    describe_import()
    describe_absolute_value()
    describe_add()
    describe_aiter()
    describe_all()
    describe_any()
    describe_ascii()
    describe_asynchronous_iterable_iterator()
    describe_awaitable()
    describe_awaitable_anext()
    describe_bin()
    describe_boolean_values()
    describe_breakpoint()
    describe_bytes_capitalize()
    describe_bytes_center()
    describe_bytes_count()
    describe_bytes_decode()
    describe_bytes_endswith()
    describe_bytes_expandtabs()
    describe_bytes_find()
    describe_bytes_index()
    describe_bytes_isalnum()
    describe_bytes_isalpha()
    describe_bytes_isascii()
    describe_bytes_isdigit()
    describe_bytes_islower()
    describe_bytes_isspace()
    describe_bytes_istitle()
    describe_bytes_isupper()
    describe_bytes_join()
    describe_bytes_ljust()
    describe_bytes_lower()
    describe_bytes_lstrip()
    describe_bytes_partition()
    describe_bytes_removeprefix()
    describe_bytes_removesuffix()
    describe_bytes_replace()
    describe_bytes_rfind()
    describe_bytes_rindex()
    describe_bytes_rjust()
    describe_bytes_rpartition()
    describe_bytes_rsplit()
    describe_bytes_rstrip()
    describe_bytes_split()
    describe_bytes_splitlines()
    describe_bytes_startswith()
    describe_bytes_strip()
    describe_bytes_swapcase()
    describe_bytes_title()
    describe_bytes_translate()
    describe_bytes_upper()
    describe_bytes_zfill()
    describe_conjugate()
    describe_memoryview_c_contiguous()
    describe_callable()
    describe_cast()
    describe_chr()
    describe_classmethod_float_fromhex()
    describe_classmethod_fromkeys()
    describe_classmethod_int_from_bytes()
    describe_clear()
    describe_code_objects()
    describe_compile()
    describe_iter()
    describe_contextmanager_enter()
    describe_contextmanager_exit()
    describe_conversion_flag_characters()
    describe_conversion_specifier_order_in_print()
    describe_conversion_types()
    describe_copy()
    describe_copyright_and_credits()
    describe_coroutines()
    describe_d_or_other()
    describe_d_or_assign_other()
    describe_d_key()
    describe_add_to_dictionary()
    describe_definition_p__name__()
    describe_definition_p__qualname__()
    describe_del_d_key()
    describe_del_start_end_step()
    describe_del_start_end()
    describe_delattr()
    describe_dict_clear()
    describe_dict_copy()
    describe_dictview_mapping()
    describe_difference()
    describe_difference_update()
    describe_dir()
    describe_discard()
    describe_divmod()
    describe_double_quotes()
    describe_ellipsis()
    describe_emscription()


if __name__ == '__main__':
    main()