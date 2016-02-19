import numpy as np
import emus

metafile = '1dwhammeta.txt'
histinfo = (-180.,180.,100)
period=360
zavar = (5,13)

emus.emus1d(metafile,histinfo,period=period,zasymptoticvar=zavar)
