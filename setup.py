#!/usr/bin/env python

from distutils.core import setup

setup(name='EMUS',
    version='0.1.0',
    description="Tools and Methods associated with the Eigenvector Method "
        "Umbrella Sampling (EMUS)",
    license='MIT',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Topic :: Scientific/Engineering'
        ],
    packages=['emus'],
    install_requires=['acor'], # Does this do what you think, Erik?

    )
