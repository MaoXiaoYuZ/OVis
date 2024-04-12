# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import ogrpc_pb2 as ogrpc__pb2


class OServiceStub(object):
    """The greeting service definition.
    !python -m grpc_tools.protoc -I. --python_out=. --pyi_out=. --grpc_python_out=. hello.proto
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Ask = channel.unary_unary(
                '/OService/Ask',
                request_serializer=ogrpc__pb2.ORequest.SerializeToString,
                response_deserializer=ogrpc__pb2.OReply.FromString,
                )


class OServiceServicer(object):
    """The greeting service definition.
    !python -m grpc_tools.protoc -I. --python_out=. --pyi_out=. --grpc_python_out=. hello.proto
    """

    def Ask(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_OServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Ask': grpc.unary_unary_rpc_method_handler(
                    servicer.Ask,
                    request_deserializer=ogrpc__pb2.ORequest.FromString,
                    response_serializer=ogrpc__pb2.OReply.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'OService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class OService(object):
    """The greeting service definition.
    !python -m grpc_tools.protoc -I. --python_out=. --pyi_out=. --grpc_python_out=. hello.proto
    """

    @staticmethod
    def Ask(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/OService/Ask',
            ogrpc__pb2.ORequest.SerializeToString,
            ogrpc__pb2.OReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
