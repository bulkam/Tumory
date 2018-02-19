# -*- coding: utf-8 -*-
"""
Created on Sat Jan 06 19:39:58 2018

@author: Mirab
"""

import os
from imtools import tools
import numpy as np


suffix = ".pklz"
imgnames = [imgname for imgname in os.listdir(os.path.dirname(os.path.abspath(__file__))) if imgname.endswith(suffix)]

sizes = list()
ranges = list()

for imgname in imgnames:
    print "   Zpracovavam obrazek "+imgname
    data, gt_mask, voxel_size = tools.load_pickle_data(imgname)
    sizes.append(voxel_size)
    ranges.extend(np.unique(data).tolist())

sizes = np.vstack(sizes)
print sizes
print np.min(sizes, axis=0)
print np.max(sizes, axis=0)
print np.unique(sizes)
print (min(ranges), max(ranges)), type(ranges[0])

