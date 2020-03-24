#
# Copyright (C) 2017 - Massachusetts Institute of Technology (MIT)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
"""
Generic imports make it easier to use this module.
"""

import inspect
from os import listdir
from os.path import join, dirname, abspath
from importlib import import_module

module_dir = abspath(dirname(__file__))

mods = set()
for module in listdir(module_dir):
    if module.endswith('.py'):
        mods.add('tsig.effects.' + module[:-3])

for module_name in mods:
    mod = import_module(module_name)
    for cls_name in dir(mod):
        obj = getattr(mod, cls_name)
        if inspect.isclass(obj): 
            globals()[cls_name] = obj


