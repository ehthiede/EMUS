-----------
Description
-----------

This is an implementation of the EMUS algorithm for recombining data 
from umbrella sampling calculations and other data sources coming from
biased probability densities.  The code is currently in open Beta,
please contact the author at `thiede [at symbol] uchicago [dot] edu` with 
any bugs, errors, or questions.

For usage and documentations, see the HTML documentation, which can be 
accessed by opening docs/html/index.html in a browser.
We are currently working on hosting the documentation online; this README
will be updated with a link once that is accomplished.

If you are using this code, please also cite the EMUS paper, which can be
found at [Journal of Chemical
Physics](http://aip.scitation.org/doi/abs/10.1063/1.4960649?journalCode=jcp).
In addition, the code for the asymtotic variance of MBAR in iter_avar.py
is based on research in the [following preprint.](https://t.co/XZwakvGI8N).
The data used in the preprint can be found [here.](https://drive.google.com/drive/folders/1BpeSdypdhxngLY9ANHV231AL1zyJ5Mmm?usp=sharing).

The code is released under the MIT license.  

Copyright (c) 2022 Erik Henning Thiede, Sherry Li, Brian Van Koten, Jonathan Weare, Aaron R. Dinner.

------------
Installation
------------

To install from the code, use the command `python setup.py install` or `pip install -e .`.  You can also install from the python repository `pip install emus`.


