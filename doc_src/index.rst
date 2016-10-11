.. emus documentation master file, created by
   sphinx-quickstart on Mon Apr  4 14:22:52 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

What is emus.py?
================

The emus.py is an implementation of the **E**\igenvector **M**\ethod for **U**\mbrella **S**\ampling (EMUS), a method for recombining statistical data from multiple biased data sources. 

This code is released under the Gnu Public License.  If you are using or modifying this code, we ask that you cite the `EMUS paper <http://scitation.aip.org/content/aip/journal/jcp/145/8/10.1063/1.4960649>`_.  This paper contains an in-depth look at the theory behind the EMUS algorithm.

Installation
------------

The emus package can be installed using pip:

>>> pip install emus

Alternatively, one can download the source code from the `GitHub page <https://github.com/ehthiede/EMUS>`_, navigate to the main directory, and run

>>> python setup.py install

to install the package.

Contents:

.. toctree::
   :glob:

   theory
   quickstart 
   datastructures
   modules/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

