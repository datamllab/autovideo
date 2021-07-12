from setuptools import setup, find_packages

PACKAGE_NAME = 'autovideo'

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

def _get_version():
    with open('autovideo/__init__.py') as f:
        for line in f:
            if line.startswith('__version__'):
                g = {}
                exec(line, g)
                return g['__version__']
        raise ValueError('`__version__` not defined')

VERSION = _get_version()

def read_file_entry_points(fname):
    with open(fname) as entry_points:
        return entry_points.read()

def merge_entry_points():
    entry_list = ['entry_points.ini']
    merge_entry = []
    for entry_name in entry_list:
        entry_point = read_file_entry_points(entry_name).replace(' ', '')
        path_list = entry_point.split('\n')[1:]
        merge_entry += path_list
    entry_point_merge = dict()
    entry_point_merge['d3m.primitives'] = list(set(merge_entry)) # remove dumplicated elements
    return entry_point_merge

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description='An Automated Video Action Recognition System',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/datamllab/autovideo",
    author='DATA Lab',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=[
        'tamu_d3m',
        'tamu_axolotl',
        'matplotlib',
        'mmcv',
   ],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points = merge_entry_points()

)
