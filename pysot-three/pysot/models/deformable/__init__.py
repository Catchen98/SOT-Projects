# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.models.deformable.DCN import DeformConvNet

deformable = {
        'DeformConvNet': DeformConvNet,
        }
def get_deformable(name,**kwargs):
    return deformable[name](**kwargs)

