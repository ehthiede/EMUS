#!/usr/bin/env python

from setuptools import setup,find_packages

setup(name='emus',
    version='0.9.2b2',
    description="Tools and Methods associated with the Eigenvector Method for Umbrella Sampling (EMUS)",
    license='GPL',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 2.7',
        ],
    packages=['emus'],
    scripts=['scripts/wemus.py'],
    install_requires=['numpy','scipy','h5py','acor'],
    url='https://github.com/ehthiede/EMUS',
    author='Erik Henning Thiede'

    )
