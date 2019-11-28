# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.models.transform.transform import Transform

transform = {
        'T_Net': Transform,
        }
def get_transform(name,**kwargs):
    return transform[name](**kwargs)

