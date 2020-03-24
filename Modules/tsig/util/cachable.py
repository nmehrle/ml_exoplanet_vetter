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
Base class for objects that can cache to disk.
"""

import os
import types
import logging
logger = logging.getLogger(__name__)


def cachable(fn):
    def _inner(self, *args, **kw):
        cacher = kw.pop('cacher', None)
        cache_key = kw.pop('cache_key', None)
        cache_log = kw.pop('cache_log', False)

        if cacher and cache_key is None:
            cache_key = cacher.generate_key(self, type(self), fn, *args, **kw)

        # get a cached object if one exists...
        if cacher is not None and cacher.has(cache_key):
            if cache_log:
                logger.debug("Getting cached version: %s" % cache_key)
            return cacher.get(cache_key)

        # ...or get the actual object...
        results = fn(self, *args, **kw)

        # ...then cache results
        if cacher is not None:
            if cache_log:
                logger.debug("Saving cached version: %s" % cache_key)
            if isinstance(results, types.GeneratorType):
                results = list(results)
            cacher.set(cache_key, results)

        return results
    return _inner


def optional_cachable(label):
    """Like cachable, but can be configured to not-cache by the configuration"""
    def _outer(fn):
        cache_call = cachable(fn)
        def _inner(self, *args, **kw):
            if 'cache_mode' not in kw or label in kw.pop('cache_mode'):
                return cache_call(self, *args, **kw)
            else:
                return fn(self, *args, **kw)
        return _inner
    return _outer


class DirectoryCacher(object):
    def __init__(self, directory=None, subdir=None, prefix=None):
        self.prefix = prefix
        if directory is None:
            directory = os.path.join(os.environ['HOME'], '.cache/tsig')
        self.directory = directory
        if subdir is not None:
            self.directory = "%s/%s" % (directory, subdir)
        logger.debug("cache to %s" % self.directory)

    def generate_key(self, obj, cls, fn, *args, **kw):
        """
        Generate a key based on the class, function and arguments.

        %(class_name)/%(prefix)_%(function)_%(args)
        """
        label = fn.__name__
        if self.prefix:
            label = "%s_%s" % (self.prefix, label)
        if hasattr(obj, 'cache_prefix'):
            label = "%s_%s" % (getattr(obj, 'cache_prefix')(), label)
        args = list(args) + ["%s=%s" % (a,str(b)) for a,b in kw.items()]
        if args:
            label = "%s_%s" % (label, "_".join([str(a) for a in args]))
        label = "%s/%s" % (cls.__name__, label)
        return label

    def get_ext(self):
        return ""

    def get_filename(self, key):
        """Returns the directory for this key"""
        return os.path.join(self.directory, key + self.get_ext())

    def has(self, key):
        return os.path.isfile(self.get_filename(key))

    def get(self, key):
        """Load the data from cache"""
        return self._read(self.get_filename(key))

    def set(self, key, data):
        """Save the data to cache"""        
        filename = self.get_filename(key)
        path = os.path.dirname(filename)
        if not os.path.isdir(path):
            os.makedirs(path)
        self._save(filename, data)

    def _read(self, filename):
        raise NotImplementedError()

    def _save(self, filename, data):
        raise NotImplementedError()


class PickleCacher(DirectoryCacher):
    def __init__(self, directory=None, subdir=None, prefix=None):
        super(PickleCacher, self).__init__(directory, subdir, prefix)

    def get_ext(self):
        return ".pkl"

    def _read(self, filename):
        import pickle
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def _save(self, filename, data):
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


class CPickleCacher(DirectoryCacher):
    def __init__(self, directory=None, subdir=None, prefix=None):
        super(CPickleCacher, self).__init__(directory, subdir, prefix)

    def get_ext(self):
        return ".cpkl"

    def _read(self, filename):
        import cPickle
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def _save(self, filename, data):
        import cPickle
        with open(filename, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


class NumpyCacher(DirectoryCacher):
    def __init__(self, directory=None, subdir=None, prefix=None):
        super(NumpyCacher, self).__init__(directory, subdir, prefix)

    def get_ext(self):
        return ".npy"

    def _save(self, filename, data):
        import numpy as np
        np.save(filename, data)

    def _read(self, filename):
        import numpy as np
        return np.load(filename)
