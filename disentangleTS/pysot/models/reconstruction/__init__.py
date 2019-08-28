# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.models.reconstruction.decoder import Decoder

RECONSTRUCTION = {
               'Reconstruction': Decoder,
               #'MReconstruction': Decoder
              }
def get_decoder(name,**kwargs):
    return RECONSTRUCTION[name](**kwargs)

