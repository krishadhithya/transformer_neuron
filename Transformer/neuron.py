import dendrite
import grpc
import numpy as np
import pickle
import threading
import random
import time
from Crypto.Hash import SHA256
from threading import Lock
from concurrent import futures
from loguru import logger

class Buffer:
    
    def __init__(self,
                 source_id=None,
                 parent_id=None,
                 message_id=None,
                 create_time=None,
                 lspikes=None,
                 uspikes=None,
                 dspikes=None,
                 lgrads=None):

        self.source_id = source_id
        self.parent_id = parent_id
        self.message_id = message_id
        self.create_time = create_time
        self.lspikes = lspikes
        self.uspikes = uspikes
        self.dspikes = dspikes
        self.lgrads = lgrads

    def set(self,
            source_id=None,
            parent_id=None,
            message_id=None,
            create_time=None,
            lspikes=None,
            uspikes=None,
            dspikes=None,
            lgrads=None):

        if not self.source_id:
            self.source_id = source_id
        if not self.parent_id:
            self.parent_id = parent_id
        if not self.message_id:
            self.message_id = message_id
        if not self.create_time:
            self.create_time = create_time
        if not self.lspikes:
            self.lspikes = lspikes
        if not self.uspikes:
            self.uspikes = uspikes
        if not self.dspikes:
            self.dspikes = dspikes
        if not self.lgrads:
            self.lgrads = lgrads

class Neuron(dendrite.proto.transformer_pb2_grpc.BittensorServicer):
    def __init__(self, config, nucleus, metagraph):
        self.config = config
        self.nucleus = nucleus
        self.metagraph = metagraph
        self._is_training = True
        self.current_stats = {
            'gs': None,
            'step': None,
            'mem': None,
            'loss': None,
            'metrics': None,
            'scores': None
        }
        
        self.lock = Lock()
        self.memory = {}
        
        # Metrics.
        self._scores = [0 for _ in range(self.config.n_children + 1)]
        
        # Init server.
        self.server_address = self.config.bind_address + ":" + self.config.port
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        dendrite.proto.transformer_pb2_grpc.add_BittensorServicer_to_server(self, self.server)
        self.server.add_insecure_port(self.server_address)
        
        self.channels = [None for _ in range(self.config.n_children)]
        self.channel_ids = [None for _ in range(self.config.n_children)]
        
        self.connect()
    
    def connect(self):
        for i in range(self.config.n_children):
            if self.channels[i] == None:
                self._set_channel(i)
    
    def _set_channel(self, i):
        for node in self.metagraph.nodes.values():
            if node.identity in self.channel_ids:
                continue
            if node.identity == self.config.identity:
                continue
            else:
                address = node.address + ":" + node.port
                self.channels[i] = grpc.insecure_channel
                self.channel_ids[i] = node.identity
                break
    
    def __del__(self):
        self.server.stop(0)
        #self._stop_training()
        logger.debug('Stopped serving Neuron at: {}.'.format(self.server_address))
    
    def serve(self):
        self.server.start()
        self._start_training()
        logger.debug('Started serving neuron at: {}'.format(self.server_address))
    
    def _spike_future(self, channel, source_id, message_id, payload):
        if channel == None:
            return None
        try:
            # Build stub and request protobuf
            stub = dendrite.proto.transformer_pb2_grpc.BittensorStub(channel)
            
            # Create spike request protobuf
            request = dendrite.proto.transformer_pb2.SpikeRequest(
                version=1.0,
                source_id=source_id,
                parent_id=self.config.identity,
                message_id=message_id,
                payload=payload)
            
            # Send TCP spike request with futures callback
            return stub.Spike.future(request)
        except Exception as e:
            logger.exception("Exception occured when sending Spike request: {}".format(e))
            return None
    
    def _fill_dspikes(self, dspikes, futures):
        while True:
            remaining_futures = sum(1 for _ in filter(None.__ne__, self.channel_ids))
            for i in range(self.config.n_children):
                if futures[i] != None:
                    if futures[i].done():
                        remaining_futures -= 1
                    try:
                        result = futures[i].result()
                        next_dspikes = pickle.loads(result.payload).reshape(-1, 128)
                        dspikes[i] = next_dspikes
                    except:
                        pass
            if remaining_futures == 0:
                break
        return dspikes
    
    def Spike(self, request, context):
        # Unpack message.
        source_id = request.source_id
        parent_id = request.parent_id
        message_id = request.message_id
        uspikes = pickle.loads(request.payload)
        if parent_id not in self._metrix:
            self._metrix[parent_id] = 0
        self._metrics[parent_id] += 1
        
        # 1. Check and build message buffer, On recursion with loops we respond 
        # with a null message if the message_id has been seen already.
        self.lock.acquire()
        try:
            # 2. Check for dups
            if message_id in self.memory:
                # Return null response
                zeros_payload = pickle.dumps(np.zeros((len(uspikes), self.config.n_embedding)), protocol=0)
                response = dendrite.proto.transformer_pb2.SpikeResponse(
                    version=1.0,
                    source_id=source_id,
                    child_id=self.config.identity,
                    message_id=message_id,
                    payload=zeros_payload
                )
                return response
            
            # 3. Build new message buffer
            msg_buffer = Buffer(source_id=source_id,
                                parent_id=parent_id,
                                message_id=message_id,
                                create_time=time.time())
            
            self.memory[message_id] = msg_buffer
        finally:
            self.lock.release()
        
        # 4. Fill downstream spikes.
        dspikes = [np.zeros((len(uspikes), 128)) for _ in range(self.config.n_children)]
        
        # 5. Inference local neuron.
        lspikes = self.nucleus.spike(uspikes, dspikes, use_synthetic=True)
        
        # 6. sink output to memory
        self.lock.acquire()
        try:
            self.memory[message_id].set(uspikes=uspikes,
                                        lspikes=lspikes,
                                        dspikes=dspikes)
        finally:
            self.lock.release()
        
        # 7. Build response.
        payload = pickle.dumps(lspikes, protocol=0)
        response = dendrite.proto.transformer_pb2.SpikeResponse(
            version=1.0,
            source_id=source_id,
            child_id=self.config.identity,
            message_id=message_id,
            payload=payload
        )
        # Return.
        return response
    
    def Grade(self, request, context):
        # Unpack request.
        source_id = request.source_id
        parent_id = request.parent_id
        message_id = request.message_id
        ugrades = pickle.loads(request.payload)
        
        # Check for lost or badly routed grades.
        if message_id not in self.memory:
            return dendrite.proto.transformer_pb2.GradeResponse(accept=True)
        
        # Get local spikes.
        mem_buffer = self.memory[message_id]
        
        # Get downstream spikes.
        dspikes = mem_buffer.dspikes
        
        # Get upstream spikes
        uspikes = mem_buffer.uspikes
        
        # Get downstream grads and local grads.
        dgrades = self.nucleus.grade(ugrades, uspikes, dspikes)
        
        # delete memory
        del self.memory[message_id]
        
        return dendrite.proto.transformer_pb2.GradeResponse(accept=True)

    def _start_training(self):
        self._is_training = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
    
    def _stop_training(self):
        self._is_training = False
        self._thread.join()

    def _run(self):
        step = 0
        last_log_step = 0
        last_log_time = time.time()
        while self._is_training:
            step += 1
            
            # 1. Next training batch
            nounce = str(random.randint(0, 1000000000))
            spikes, target_seqs_in, target_seqs_out = self.nucleus.next_batch()
            
            # 2. Encode nounce and source
            source_id = self.config.identity
            nounce_bytes = bytes(nounce, 'utf-8')
            source_bytes = bytes(source_id, 'utf-8')
            payload_bytes = pickle.dumps(spikes, protocol=0)
            
            # 3. Create unique message hash.
            hash = SHA256.new()
            hash.update(nounce_bytes)
            hash.update(source_bytes)
            hash.update(payload_bytes)
            message_id = hash.digest()
            
            # 4. Make recursive calls to downstream neighbors.
            # Futures is a list of callbacks from each downstream call. 
            futures = []
            for channel in self.channels:
                futures.append(self._spike_future(channel, source_id, message_id, payload_bytes))
            
            # 5. Fill responses
            dspikes = [np.zeros((self.config.batch_size, self.config.n_embedding)) for _ in range(self.config.n_children)]
            dspikes = self._fill_dspikes(dspikes, futures)
            
            # 6. Train local model
            logger.info("Training local model...")
            
            
        