#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import re

import setuptools

with io.open('src/dicomslide/__init__.py', 'rt', encoding='utf8') as f:
    version = re.search(r'__version__ = \'(.*?)\'', f.read()).group(1)


setuptools.setup(
    name='dicomslide',
    version=version,
    description='High-level DICOM abstractions.',
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
        'highdicom>=0.15.2',
        'dicomweb_client>=0.55.1'
    ],
)
