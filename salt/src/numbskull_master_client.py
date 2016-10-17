"""TODO."""

from __future__ import absolute_import
from salt.utils.async import SyncWrapper
from salt.transport.client import AsyncChannel
from salt.transport.tcp import SaltMessageClient

import msgpack
import socket
import weakref
import logging

# Import Salt Libs
import salt.payload
import salt.exceptions
import salt.ext.six as six
from salt.exceptions import SaltReqTimeoutError, SaltClientError

# Import Tornado Libs
import tornado
import tornado.ioloop
import tornado.gen

# pylint: disable=import-error,no-name-in-module
if six.PY2:
    import urlparse
else:
    import urllib.parse as urlparse
# pylint: enable=import-error,no-name-in-module

log = logging.getLogger(__name__)


class InfLearn_Channel(object):
    """TODO."""

    @staticmethod
    def factory(opts, **kwargs):
        """TODO."""
        return InfLearn_ReqChannel.factory(opts, **kwargs)


class InfLearn_ReqChannel(object):
    """Factory to create Sync communication channels to the ReqServer."""

    @staticmethod
    def factory(opts, **kwargs):
        """TODO."""
        # All Sync interfaces are just wrappers around the Async ones
        sync = SyncWrapper(InfLearn_AsyncChannel.factory, (opts,), kwargs)
        return sync

    def send(self, load, tries=3, timeout=60, raw=False):
        """Send "load" to the master."""
        raise NotImplementedError()


class InfLearn_AsyncChannel(AsyncChannel):
    """Factory to create Async comm. channels to the ReqServer."""

    @classmethod
    def factory(cls, opts, **kwargs):
        """TODO."""
        if not cls._resolver_configured:
            AsyncChannel._config_resolver()
        return InfLearn_AsyncTCPChannel(opts, **kwargs)

    def send(self, load, tries=3, timeout=60, raw=False):
        """Send 'load' to the minion."""
        raise NotImplementedError()


class InfLearn_AsyncTCPChannel(InfLearn_ReqChannel):
    """
    Encapsulate sending routines to tcp.

    Note: this class returns a singleton
    """

    # This class is only a singleton per minion/master pair
    # mapping of io_loop -> {key -> channel}
    instance_map = weakref.WeakKeyDictionary()

    def __new__(cls, opts, **kwargs):
        """Only create one instance of channel per __key()."""
        # do we have any mapping for this io_loop
        io_loop = kwargs.get('io_loop') or tornado.ioloop.IOLoop.current()
        if io_loop not in cls.instance_map:
            cls.instance_map[io_loop] = weakref.WeakValueDictionary()
        loop_instance_map = cls.instance_map[io_loop]

        key = cls.__key(opts, **kwargs)
        if key not in loop_instance_map:
            log.debug('Initializing new InfLearn_AsyncTCPChannel '
                      'for {0}'.format(key))
            # we need to make a local variable for this, as we are going to
            # store it in a WeakValueDictionary-- which will remove the item
            # if no one references it-- this forces a reference while we
            # return to the caller
            new_obj = object.__new__(cls)
            new_obj.__singleton_init__(opts, **kwargs)
            loop_instance_map[key] = new_obj
        else:
            log.debug('Re-using AsyncTCPReqChannel for {0}'.format(key))
        return loop_instance_map[key]

    @classmethod
    def __key(cls, opts, **kwargs):
        if 'minion_uri' in kwargs:
            opts['minion_uri'] = kwargs['minion_uri']
        return (opts['master_uri'])

    # must be empty for singletons, since __init__ will *always* be called
    def __init__(self, opts, **kwargs):
        """TODO."""
        pass

    # an init for the singleton instance to call
    def __singleton_init__(self, opts, **kwargs):
        """TODO."""
        self.opts = dict(opts)

        self.serial = salt.payload.Serial(self.opts)
        self.io_loop = kwargs.get('io_loop') or tornado.ioloop.IOLoop.current()
        resolver = kwargs.get('resolver')

        parse = urlparse.urlparse(self.opts['minion_uri'])
        host, port = parse.netloc.rsplit(':', 1)
        self.minion_addr = (host, int(port))
        self._closing = False
        self.message_client = SaltMessageClient(
            self.opts, host, int(port), io_loop=self.io_loop,
            resolver=resolver)

    def close(self):
        """TODO."""
        if self._closing:
            return
        self._closing = True
        self.message_client.close()

    def __del__(self):
        """TODO."""
        self.close()

    def _package_load(self, load):
        """TODO."""
        return {'load': load}

    @tornado.gen.coroutine
    def _transfer(self, load, tries=3, timeout=60):
        """TODO."""
        ret = yield self.message_client.send(self._package_load(load),
                                             timeout=timeout)
        raise tornado.gen.Return(ret)

    @tornado.gen.coroutine
    def send(self, load, tries=3, timeout=60, raw=False):
        """
        Send a request.

        Returns a future which will complete when we send the message
        """
        try:
            ret = yield self._transfer(load, tries=tries, timeout=timeout)
        except tornado.iostream.StreamClosedError:
            # Convert to 'SaltClientError' so that clients can handle this
            # exception more appropriately.
            raise SaltClientError('Connection to minion lost')
        raise tornado.gen.Return(ret)
