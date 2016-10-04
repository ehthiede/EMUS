#!/usr/bin/env python

from setuptools import setup,find_packages

setup(name='EMUS',
    version='0.1.0',
    description="Tools and Methods associated with the Eigenvector Method "
        "Umbrella Sampling (EMUS)",
    license='MIT',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Topic :: Scientific/Engineering'
        ],
    packages=find_packages(),
    install_requires=['numpy','hdf5','acor'], # Does this do what you think, Erik?

    )
