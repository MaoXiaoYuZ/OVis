

from concurrent import futures
import logging
import pickle

import grpc
import ogrpc_pb2
import ogrpc_pb2_grpc


class OService(ogrpc_pb2_grpc.OServiceServicer):
    def Ask(self, request, context):
        request_obj = pickle.loads(request.pkl)
        reply_obj = {'reply': "You Ask me!"}
        # import numpy as np
        # reply_obj = np.random.rand(100000, 3).astype('float16')
        return ogrpc_pb2.OReply(pkl=pickle.dumps(reply_obj))
        # return ogrpc_pb2.OReply(pkl=reply_obj.tobytes())


def serve():
    port = "50051"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    ogrpc_pb2_grpc.add_OServiceServicer_to_server(OService(), server)
    server.add_insecure_port("[::]:" + port)
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig()
    serve()
