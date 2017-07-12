#!/usr/bin/env python

from setuptools import setup,find_packages

setup(name='emus',
    version='0.9.2b1',
    description="Tools and Methods associated with the Eigenvector Method "
        "Umbrella Sampling (EMUS)",
    license='GPL',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Topic :: Scientific/Engineering'
        ],
#    packages=find_packages(),
#    package_dir=['emus'],
    packages=['emus'],
    scripts=['scripts/wemus.py'],
    install_requires=['numpy','scipy','h5py','acor'],
#    install_requires=['numpy','scipy'],

    )
