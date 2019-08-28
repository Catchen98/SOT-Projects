# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.models.discriminate.discriminator import Discriminator

DISCRIMINATOR = {
               'Discriminator': Discriminator,
              }
def get_discriminator(name, **kwargs):
    return DISCRIMINATOR[name](**kwargs)

