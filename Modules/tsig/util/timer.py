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
Timer object with logging.
"""

import logging
import time


class Timer(object):
    def __init__(self, filename):
        self.t0 = 0
        self.t1 = 0
        hand = logging.FileHandler(filename)
        fmt = '%(asctime)s %(message)s'
        datefmt = '%Y.%m.%d %H:%M:%S'
        hand.setFormatter(logging.Formatter(fmt, datefmt))
        self.logger = logging.getLogger("timer")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        self.logger.addHandler(hand)
        self.start()

    def start(self):
        self.t0 = time.time()

    def stop(self):
        self.t1 = time.time()

    def elapsed(self, use_stop=True):
        if use_stop:
            return self.t1 - self.t0
        return time.time() - self.t0

    def elapsed_fmt(self, use_stop=True, fmt='seconds'):
        return "%.3f seconds" % self.elapsed(use_stop)

    def info(self, msg):
        self.logger.info(msg)

    def report(self):
        self.info("elapsed: %.3f start: %.3f stop: %.3f" % (
            self.elapsed(), self.t0, self.t1))
