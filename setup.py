#!/usr/bin/env python

from setuptools import setup, find_packages
from glob import glob
from os.path import basename, splitext

setup(name='emus',
      version='0.9.4',
      description="Tools and Methods associated with the Eigenvector Method for Umbrella Sampling (EMUS)",
      license='LGPL',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Topic :: Scientific/Engineering',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
      ],
      # packages=['emus'],
      packages=find_packages('src'),
      package_dir={'': 'src'},
      py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
      scripts=['scripts/wemus.py'],
      install_requires=['numpy', 'scipy', 'h5py'],
      url='https://github.com/ehthiede/EMUS',
      author='Erik Henning Thiede'

      )
