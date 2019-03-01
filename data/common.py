#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import re
#
# def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
#     assert os.path.isdir(directory), 'dataset is not exists!{}'.format(directory)
#
#     return sorted([os.path.join(root, f)
#                    for root, _, files in os.walk(directory) for f in files
#                    if re.match(r'([\w]+\.(?:' + ext + '))', f)])
# #
def list_pictures(path):
    pic_list = os.listdir(path)
    return pic_list
