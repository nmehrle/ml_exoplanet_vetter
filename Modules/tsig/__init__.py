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

"""
Module base for the tsig python module.
"""

__version__ = '0.41'
__pkgname__ = 'tess-tsig'

import locale
import logging
import platform
import sys

# FIXME: ensure this does not mess with log config customizations
# FIXME: make this work for tsig as a library, not just standalone app
# FIXME: include thread name in the log
# FIXME: ensure logging works with chelsea's multiple/distributed process usage

def platform_info():
    lines = []
    lines.append('TSIG: %s' % __version__)
    lines.append('Python: %s' % sys.version.replace('\n', ' '))
    lines.append('Host: %s' % platform.node())
    lines.append('Platform: %s' % platform.platform())
    lines.append('Locale: %s' % locale.setlocale(locale.LC_ALL))
    return lines

def setup_logging(debug=False, filename=None):
    """Provide a sane set of defaults for logging."""
    level = logging.DEBUG if debug else logging.INFO
    if filename is None:
        filename = '-'

    if filename == '-':
        hand = logging.StreamHandler()
    else:
        hand = logging.FileHandler(filename)

    fmt = '%(asctime)s %(levelname)s %(funcName)s: %(message)s' if level == logging.DEBUG else '%(asctime)s %(message)s'
    datefmt = '%Y.%m.%d %H:%M:%S'
    hand.setFormatter(logging.Formatter(fmt, datefmt))

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers = []
    root_logger.addHandler(hand)

    for line in platform_info():
        logging.debug(line)
