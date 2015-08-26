from astropy.io import fits
import numpy as np
import os



marzDir = r"C:\Users\Samuel\Google Drive\Uni\ENGG4801\ADASS\Analysis\combined_2\marz_interpAndIndex"
fitsInputDir = r"C:\Users\Samuel\Desktop\New folder (3)"
fitsoutputDir = r"C:\Users\Samuel\Desktop\output"

results = {}

for filename in os.listdir(marzDir):
    with open(os.path.join(marzDir, filename), 'r') as f:
        qop4s = []
        for line in f:
            if line.startswith("#"): continue
            i = line.split(",")
            qop = int(i[13])
            if qop == 4 or qop == 6:
                qop4s.append(i[1].strip())
        results[filename[:filename.index("_SRH")]] = qop4s

for filename in os.listdir(fitsInputDir):
    key = filename[:filename.index(".fits")]
    res = results.get(key)
    if res is None:
        print("Generate results for %s" % filename)
    else:
        fn = filename
        if len(res) == 0:
            continue
        print(filename)
        hdulist = fits.open(os.path.join(fitsInputDir, filename), mode='update', scale_back=False)
        toKeep = np.array([ind for ind, x in enumerate(hdulist['FIBRES'].data['NAME']) if x in res])
        
        hdulist['PRIMARY'].data = hdulist['PRIMARY'].data[toKeep, :]
        hdulist['VARIANCE'].data = hdulist['VARIANCE'].data[toKeep, :]
        hdulist['FIBRES'].data = hdulist['FIBRES'].data[toKeep]

        hdulist['FIBRES'].update()
        
        for tup in hdulist.info(False):
            if tup[1] not in ['PRIMARY', 'VARIANCE', "FIBRES"]:
                del hdulist[tup[1]]
        
            
        hdulist.writeto(os.path.join(fitsoutputDir, fn))
        hdulist.close()
        


