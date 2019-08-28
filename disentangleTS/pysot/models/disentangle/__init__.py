# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.models.disentangle.disentangle import Multi_Split_fb

DISENTANGLE = {
               'Multi_Split_fb': Multi_Split_fb
              }
def get_disentangle(name, **kwargs):
    return DISENTANGLE[name](**kwargs)

