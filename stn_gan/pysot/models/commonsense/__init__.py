# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.models.commonsense.commonsense import Explicit_corr

COMMONSENSE = {
               'Explicit_corr': Explicit_corr
              }
def get_commonsense_head(name,**kwargs):
    return COMMONSENSE[name](**kwargs)

