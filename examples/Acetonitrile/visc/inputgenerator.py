"""Created on Mon Jul 13 14:45:35 2015

@author: mhumbert
"""

import os
import random

samplefile = open("test_1/in.visc").readlines()
filelist = range(2, 51)
for num in filelist:
    os.system(f"mkdir test_{num}")
    output = open(f"test_{num}/in.visc", "w")
    for line in range(len(samplefile)):
        if line == 31:
            output.write(
                f"velocity        all create  ${{mytemp}} {random.randint(1, 999999999)} units box \n"
            )
        else:
            output.write(samplefile[line])
    output.close()
    os.system(f"cp test_1/mol.data test_{num}/")
