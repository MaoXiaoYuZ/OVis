from __future__ import print_function

import logging
import pickle
import numpy as np

import grpc
import ogrpc_pb2
import ogrpc_pb2_grpc


def oconnect(url):
    global channel
    channel = grpc.insecure_channel(url)

# @profile
def oask(request_obj):
    global channel
    stub = ogrpc_pb2_grpc.OServiceStub(channel)
    response = stub.Ask(ogrpc_pb2.ORequest(pkl=pickle.dumps(request_obj)))
    # return np.frombuffer(response.pkl, dtype='float16').reshape(100000, 3)
    return pickle.loads(response.pkl)

def oclose():
    global channel
    if channel:
        channel.close()
        channel = None

@profile
def test():
    # oconnect("localhost:50051")
    oconnect("59.77.18.8:50051")
    for i in range(10):
        response = oask(np.random.rand(4, 256, 3))
        print(response.shape)
    oclose()

if __name__ == "__main__":
    logging.basicConfig()
    test()

