# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.models.commonsense.commonsense import Explicit_corr,Explicit_corr1,Explicit_corr2

COMMONSENSE = {
               'Explicit_corr': Explicit_corr,
               'Explicit_corr1': Explicit_corr1,
               'Explicit_corr2': Explicit_corr2,
              }
def get_commonsense(name):
    return COMMONSENSE[name]()

