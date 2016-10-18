"""TODO."""

# Import python libs
from __future__ import absolute_import
import logging
import sys
import os
import time
import pydoc
import urlparse
import traceback

# libs for server
import msgpack
import socket
import errno
import signal

# Import salt libs
import salt.utils.event
import salt.utils
import salt.payload
import salt.exceptions
import salt.transport.frame
import salt.ext.six as six
from salt.exceptions import SaltReqTimeoutError, SaltClientError
from salt.utils.process import default_signals, \
                               SignalHandlingMultiprocessingProcess

# Import Tornado Libs
import tornado
import tornado.tcpserver
import tornado.gen
import tornado.concurrent
import tornado.tcpclient
import tornado.netutil
import tornado.ioloop
LOOP_CLASS = tornado.ioloop.IOLoop

USE_LOAD_BALANCER = False

if USE_LOAD_BALANCER:
    import threading
    import multiprocessing
    import errno
    import tornado.util
    from salt.utils.process import SignalHandlingMultiprocessingProcess

# pylint: disable=import-error,no-name-in-module
if six.PY2:
    import urlparse
else:
    import urllib.parse as urlparse
# pylint: enable=import-error,no-name-in-module

log = logging.getLogger(__name__)


def _set_tcp_keepalive(sock, opts):
    """Ensure that TCP keepalives are set for the socket."""
    if hasattr(socket, 'SO_KEEPALIVE'):
        if opts.get('tcp_keepalive', False):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            if hasattr(socket, 'SOL_TCP'):
                if hasattr(socket, 'TCP_KEEPIDLE'):
                    tcp_keepalive_idle = opts.get('tcp_keepalive_idle', -1)
                    if tcp_keepalive_idle > 0:
                        sock.setsockopt(
                            socket.SOL_TCP, socket.TCP_KEEPIDLE,
                            int(tcp_keepalive_idle))
                if hasattr(socket, 'TCP_KEEPCNT'):
                    tcp_keepalive_cnt = opts.get('tcp_keepalive_cnt', -1)
                    if tcp_keepalive_cnt > 0:
                        sock.setsockopt(
                            socket.SOL_TCP, socket.TCP_KEEPCNT,
                            int(tcp_keepalive_cnt))
                if hasattr(socket, 'TCP_KEEPINTVL'):
                    tcp_keepalive_intvl = opts.get('tcp_keepalive_intvl', -1)
                    if tcp_keepalive_intvl > 0:
                        sock.setsockopt(
                            socket.SOL_TCP, socket.TCP_KEEPINTVL,
                            int(tcp_keepalive_intvl))
            if hasattr(socket, 'SIO_KEEPALIVE_VALS'):
                # Windows doesn't support TCP_KEEPIDLE, TCP_KEEPCNT, nor
                # TCP_KEEPINTVL. Instead, it has its own proprietary
                # SIO_KEEPALIVE_VALS.
                tcp_keepalive_idle = opts.get('tcp_keepalive_idle', -1)
                tcp_keepalive_intvl = opts.get('tcp_keepalive_intvl', -1)
                # Windows doesn't support changing something equivalent to
                # TCP_KEEPCNT.
                if tcp_keepalive_idle > 0 or tcp_keepalive_intvl > 0:
                    # Windows defaults may be found by using the link below.
                    # Search for 'KeepAliveTime' and 'KeepAliveInterval'.
                    # https://technet.microsoft.com/en-us/library/bb726981.aspx#EDAA
                    # If one value is set and the other isn't, we still need
                    # to send both values to SIO_KEEPALIVE_VALS and they both
                    # need to be valid. So in that case, use the Windows
                    # default.
                    if tcp_keepalive_idle <= 0:
                        tcp_keepalive_idle = 7200
                    if tcp_keepalive_intvl <= 0:
                        tcp_keepalive_intvl = 1
                    # The values expected are in milliseconds, so multiply by
                    # 1000.
                    sock.ioctl(socket.SIO_KEEPALIVE_VALS, (
                        1, int(tcp_keepalive_idle * 1000),
                        int(tcp_keepalive_intvl * 1000)))
        else:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 0)

if USE_LOAD_BALANCER:
    class LoadBalancerServer(SignalHandlingMultiprocessingProcess):
        """
        This is a TCP server.

        Raw TCP server which runs in its own process and will listen
        for incoming connections. Each incoming connection will be
        sent via multiprocessing queue to the workers.
        Since the queue is shared amongst workers, only one worker will
        handle a given connection.
        """

        # TODO: opts!
        # Based on default used in tornado.netutil.bind_sockets()
        backlog = 128

        def __init__(self, opts, socket_queue, log_queue=None):
            """TODO."""
            super(LoadBalancerServer, self).__init__(log_queue=log_queue)
            self.opts = opts
            self.socket_queue = socket_queue
            self._socket = None

        # __setstate__ and __getstate__ are only used on Windows.
        # We do this so that __init__ will be invoked on Windows in the child
        # process so that a register_after_fork() equivalent will work on
        # Windows.
        def __setstate__(self, state):
            """TODO."""
            self._is_child = True
            self.__init__(
                state['opts'],
                state['socket_queue'],
                log_queue=state['log_queue']
            )

        def __getstate__(self):
            """TODO."""
            return {'opts': self.opts,
                    'socket_queue': self.socket_queue,
                    'log_queue': self.log_queue}

        def close(self):
            """TODO."""
            if self._socket is not None:
                self._socket.shutdown(socket.SHUT_RDWR)
                self._socket.close()
                self._socket = None

        def __del__(self):
            """TODO."""
            self.close()

        def run(self):
            """Start the load balancer."""
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            _set_tcp_keepalive(self._socket, self.opts)
            self._socket.setblocking(1)
            self._socket.bind((self.opts['inf_learn_interface'],
                               int(self.opts['inf_learn_port'])))
            self._socket.listen(self.backlog)

            while True:
                try:
                    # Wait for a connection to occur since the socket is
                    # blocking.
                    connection, address = self._socket.accept()
                    # Wait for a free slot to be available to put
                    # the connection into.
                    # Sockets are picklable on Windows in Python 3.
                    self.socket_queue.put((connection, address), True, None)
                except socket.error as e:
                    # ECONNABORTED indicates that there was a connection
                    # but it was closed while still in the accept queue.
                    # (observed on FreeBSD).
                    if tornado.util.errno_from_exception(e) == \
                       errno.ECONNABORTED:
                        continue
                    raise


class InfLearnMessageServer(tornado.tcpserver.TCPServer, object):
    """
    This is a raw TCP server.

    Raw TCP server which will receive all of the TCP streams and re-assemble
    messages that are sent through to us
    """

    def __init__(self, message_handler, logger, *args, **kwargs):
        """TODO."""
        super(InfLearnMessageServer, self).__init__(*args, **kwargs)

        self.clients = []
        self.message_handler = message_handler
        self.log = logger
        self.log.debug('Inside InfLearnMessageServer')

    @tornado.gen.coroutine
    def handle_stream(self, stream, address):
        """Handle incoming streams and add messages to the incoming queue."""
        self.log.debug('InfLearn client {0} connected'.format(address))
        self.clients.append((stream, address))
        unpacker = msgpack.Unpacker()
        try:
            while True:
                wire_bytes = yield stream.read_bytes(4096, partial=True)
                unpacker.feed(wire_bytes)
                for framed_msg in unpacker:
                    if six.PY3:
                        framed_msg = salt.transport.frame.decode_embedded_strs(
                            framed_msg
                        )
                    header = framed_msg['head']
                    self.io_loop.spawn_callback(self.message_handler, stream,
                                                header, framed_msg['body'])

        except tornado.iostream.StreamClosedError:
            self.log.debug('InfLearn client disconnected {0}'.format(address))
            self.clients.remove((stream, address))
        except Exception as e:
            self.log.debug('Other minion-side InfLearn '
                           'exception: {0}'.format(e))
            self.clients.remove((stream, address))
            stream.close()

    def shutdown(self):
        """Shutdown the whole server."""
        for item in self.clients:
            client, address = item
            client.close()
            self.clients.remove(item)

if USE_LOAD_BALANCER:
    class LoadBalancerWorker(InfLearnMessageServer):
        """
        This receives TCP connections.

        This will receive TCP connections from 'LoadBalancerServer' via
        a multiprocessing queue.
        Since the queue is shared amongst workers, only one worker will handle
        a given connection.
        """

        def __init__(self, socket_queue, message_handler, logger, *args,
                     **kwargs):
            """TODO."""
            super(LoadBalancerWorker, self).__init__(
                message_handler, logger, *args, **kwargs)
            self.socket_queue = socket_queue

            t = threading.Thread(target=self.socket_queue_thread)
            t.start()

        def socket_queue_thread(self):
            """TODO."""
            try:
                while True:
                    client_socket, address = self.socket_queue.get(True, None)

                    # 'self.io_loop' initialized in super class
                    # 'tornado.tcpserver.TCPServer'.
                    # 'self._handle_connection' defined in same super class.
                    self.io_loop.spawn_callback(
                        self._handle_connection, client_socket, address)
            except (KeyboardInterrupt, SystemExit):
                pass


class TCPReqServerMinionChannel(object):
    """TODO."""

    backlog = 5

    def __init__(self, logger, opts, salt):
        """TODO."""
        self.log = logger
        self._opts = opts
        self._socket = None
        self._salt = salt

    @property
    def socket(self):
        """TODO."""
        return self._socket

    def close(self):
        """TODO."""
        if self._socket is not None:
            try:
                self._socket.shutdown(socket.SHUT_RDWR)
            except socket.error as exc:
                if exc.errno == errno.ENOTCONN:
                    # We may try to shutdown a socket which is already
                    # disconnected.
                    # Ignore this condition and continue.
                    pass
                else:
                    raise exc
            self._socket.close()
            self._socket = None

    def __del__(self):
        """TODO."""
        self.close()

    def pre_fork(self, process_manager):
        """Pre-fork we need to initialize socket."""
        if USE_LOAD_BALANCER:
            self.socket_queue = multiprocessing.Queue()
            process_manager.add_process(
                LoadBalancerServer, args=(self._opts, self.socket_queue)
            )
        else:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            _set_tcp_keepalive(self._socket, self._opts)
            self._socket.setblocking(0)
            self._socket.bind((self._opts['inf_learn_interface'],
                               int(self._opts['inf_learn_port'])))

    def post_fork(self, payload_handler, io_loop):
        """
        TODO.

        After forking we need to create all of the local sockets to listen to
        the router
        payload_handler: function to call with your payloads
        """
        self.payload_handler = payload_handler
        self.io_loop = io_loop
        self.serial = salt.payload.Serial(self._opts)
        if USE_LOAD_BALANCER:
            self.req_server = LoadBalancerWorker(
                self.socket_queue, self.handle_message, \
                self.log, io_loop=self.io_loop
            )
        else:
            self.req_server = InfLearnMessageServer(self.handle_message,
                                                    self.log,
                                                    io_loop=self.io_loop)
            self.req_server.add_socket(self._socket)
            self._socket.listen(self.backlog)

    def fire_local_event(self, payload):
        """TODO."""
        try:
            tag = payload['load']['tag']
            data = payload['load']['data']
            self._salt['event.fire'](data, tag)
            return True
        except:
            return False

    def handle_message(self, stream, header, payload):
        """Handle incoming messages from underylying tcp streams."""
        if self.fire_local_event(payload):
            try:
                stream.write(salt.transport.frame.frame_msg('OK',
                                                            header=header))
            except:
                raise tornado.gen.Return()
        else:
            try:
                stream.write(salt.transport.frame.frame_msg('ERROR',
                                                            header=header))
            except:
                raise tornado.gen.Return()


class InfLearnMinionServer(object):
    """TODO."""

    def __init__(self, opts, logger, salt, log_queue=None):
        """TODO."""
        self.opts = opts
        self.log_queue = log_queue
        self.log = logger
        self.salt = salt

    def __bind(self):
        """TODO."""
        if self.log_queue is not None:
            salt.log.setup.set_multiprocessing_logging_queue(self.log_queue)
        salt.log.setup.setup_multiprocessing_logging(self.log_queue)

        dfn = os.path.join(self.opts['cachedir'], '.dfn')
        if os.path.isfile(dfn):
            try:
                if salt.utils.is_windows() and not os.access(dfn, os.W_OK):
                    # Cannot delete read-only files on Windows.
                    os.chmod(dfn, stat.S_IRUSR | stat.S_IWUSR)
                os.remove(dfn)
            except os.error:
                pass

        self.process_manager = salt.utils.process.ProcessManager(
                                   name='ReqMinionInfLearnServer_PM'
                               )

        req_channels = []
        tcp_only = True
        chan = TCPReqServerMinionChannel(self.log, self.opts, self.salt)
        chan.pre_fork(self.process_manager)
        req_channels.append(chan)
        # Reset signals to default ones before adding processes to the process
        # manager. We don't want the processes being started to inherit those
        # signal handlers
        kwargs = {}
        with default_signals(signal.SIGINT, signal.SIGTERM):
            for ind in range(int(self.opts['inf_learn_threads'])):
                name = 'InfLearnWorker-{0}'.format(ind)
                self.process_manager.add_process(InfLearnWorker,
                                                 args=(self.opts,
                                                       req_channels,
                                                       name,
                                                       self.log),
                                                 kwargs=kwargs,
                                                 name=name)
        self.process_manager.run()

    def run(self):
        """Start up the InfLearnServer."""
        self.__bind()

    def destroy(self, signum=signal.SIGTERM):
        """TODO."""
        if hasattr(self, 'process_manager'):
            self.process_manager.stop_restarting()
            self.process_manager.send_signal_to_processes(signum)
            self.process_manager.kill_children()

    def __del__(self):
        """TODO."""
        self.destroy()


class InfLearnWorker(SignalHandlingMultiprocessingProcess):
    """
    Manages backend operations.

    The worker multiprocess instance to manage the backend operations for the
    minion during inference and learning.
    """

    def __init__(self,
                 opts,
                 req_channels,
                 name,
                 logger,
                 **kwargs):
        """
        Create a salt minion inference learning worker process.

        :param dict opts: The salt options

        :rtype: InfLearngWorker
        :return: Inference Learning worker
        """
        kwargs['name'] = name
        SignalHandlingMultiprocessingProcess.__init__(self, **kwargs)
        self.opts = opts
        self.log = logger
        self.req_channels = req_channels

        self.k_mtime = 0

    # We need __setstate__ and __getstate__ to also pickle 'SMaster.secrets'.
    # Otherwise, 'SMaster.secrets' won't be copied over to the spawned process
    # on Windows since spawning processes on Windows requires pickling.
    # These methods are only used when pickling so will not be used on
    # non-Windows platforms.
    def __setstate__(self, state):
        """TODO."""
        self._is_child = True
        SignalHandlingMultiprocessingProcess.__init__(
                                                 self,
                                                 log_queue=state['log_queue']
                                             )
        self.opts = state['opts']
        self.req_channels = state['req_channels']
        self.k_mtime = state['k_mtime']

    def __getstate__(self):
        """TODO."""
        return {'opts': self.opts,
                'req_channels': self.req_channels,
                'k_mtime': self.k_mtime,
                'log_queue': self.log_queue}

    def _handle_signals(self, signum, sigframe):
        """TODO."""
        for channel in getattr(self, 'req_channels', ()):
            channel.close()
        super(InfLearnWorker, self)._handle_signals(signum, sigframe)

    def __bind(self):
        """Bind to the local port."""
        self.io_loop = LOOP_CLASS()
        self.io_loop.make_current()
        for req_channel in self.req_channels:
            req_channel.post_fork(self._handle_payload, io_loop=self.io_loop)
        self.log.debug('Inside worker ' + self.name)
        try:
            self.io_loop.start()
        except (KeyboardInterrupt, SystemExit):
            # Tornado knows what to do
            pass

    @tornado.gen.coroutine
    def _handle_payload(self, payload):
        """
        TODO.

        The _handle_payload method is the key method used to figure out what
        needs to be done with communication to the server
        """
        raise tornado.gen.Return(payload)

    def run(self):
        """Start a Minion Inference Learning Worker."""
        salt.utils.appendproctitle(self.name)
        self.__bind()


####################
# ENGINE MAIN LOOP #
####################

def start():
    """TODO."""
    log.debug('Starting Numbskull Minion InfLearn Server')
    ilServer = InfLearnMinionServer(__opts__, log, __salt__)
    ilServer.run()
