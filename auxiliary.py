"""
Prep the original data !
"""
import os
import pickle

import numpy as np
import pandas as pd

from dev_norpy_config import RESOURCES_DIR
from collections import OrderedDict

#import moments

moments_import = pickle.load(open(os.path.join(RESOURCES_DIR,"moments.respy.pkl"),"rb"))

weigthing_import = pickle.load(open(os.path.join(RESOURCES_DIR,"weighing.respy.pkl"),"rb"))

#transform both objects
#moments
moments_final = OrderedDict()

moments_final["Choice Probability"] = OrderedDict(
    {x:[moments_import["Choice Probability"][x][0]] +
       moments_import["Choice Probability"][x][2:4]
     for x in moments_import["Choice Probability"].keys()})

#moments_final["Wage Distribution"] = OrderedDict(
#    {x:moments_import["Wage Distribution"][x][0] for x in moments_import["Wage Distribution"].keys() }
#)
moments_final["Wage Distribution"] = moments_import["Wages Distribution"]

moments_final["Final Schooling"] = moments_import["Final Schooling"]

#weighting
weigthing_final = weigthing_import
