#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import re
from pathlib import Path

import setuptools

root_directory = Path(__file__).parent
readme_filepath = root_directory.joinpath('README.md')
long_description = readme_filepath.read_text()

version_filepath = root_directory.joinpath('src/dicomslide/version.py')
with io.open(version_filepath, 'rt', encoding='utf8') as f:
    version = re.search(r'__version__ = \'(.*?)\'', f.read()).group(1)


setuptools.setup(
    name='dicomslide',
    version=version,
    description=(
        'Library for reading whole slide images and derived information '
        'in DICOM format.'
    ),
    long_description=long_description,
    author='Markus D. Herrmann',
    maintainer='Markus D. Herrmann',
    url='https://github.com/herrmannlab/dicomslide',
    license='MIT',
    platforms=['Linux', 'MacOS', 'Windows'],
    classifiers=[
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Science/Research',
        'Topic :: Multimedia :: Graphics',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Development Status :: 4 - Beta',
    ],
    include_package_data=True,
    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'},
    python_requires='>=3.6',
    install_requires=[
        'dicomweb_client>=0.56.2',
        'highdicom>=0.16.0',
        'scipy>=1.0',
    ],
)
