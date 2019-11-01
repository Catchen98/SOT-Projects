# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.models.stn.STN import STNNet

STN = {
        'STNNet': STNNet,
        #'MReconstruction': Decoder
        }
def get_stn(name):
    return STN[name]()

