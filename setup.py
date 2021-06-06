"""Pip installation script for `matflow`."""

import os
import re
from setuptools import find_packages, setup


def get_version():

    ver_file = 'matflow/_version.py'
    with open(ver_file) as handle:
        ver_str_line = handle.read()

    ver_pattern = r'^__version__ = [\'"]([^\'"]*)[\'"]'
    match = re.search(ver_pattern, ver_str_line, re.M)
    if match:
        ver_str = match.group(1)
    else:
        msg = 'Unable to find version string in "{}"'.format(ver_file)
        raise RuntimeError(msg)

    return ver_str


def get_long_description():

    readme_file = 'README.md'
    with open(readme_file, encoding='utf-8') as handle:
        contents = handle.read()

    return contents


package_data = [
    os.path.join(*os.path.join(root, f).split(os.path.sep)[1:])
    for root, dirs, files in os.walk(os.path.join('matflow', 'data'))
    for f in files
]

setup(
    name='matflow',
    version=get_version(),
    description=('Computational workflow management for materials science.'),
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    author='Adam J. Plowman',
    author_email='adam.plowman@manchester.ac.uk',
    packages=find_packages(),
    package_data={
        'matflow': package_data,
    },
    install_requires=[
        'matflow-demo-extension',
        'hpcflow>=0.1.16',
        'click>7.0',
        'hickle>=4.0.1',
        'ruamel.yaml',
        'numpy',
        'pyperclip',
        'black',
        'autopep8',
    ],
    project_urls={
        'Github': 'https://github.com/Lightform-group/matflow',
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
        'Operating System :: OS Independent',
    ],
    entry_points="""
        [console_scripts]
        matflow=matflow.cli:cli
    """
)
