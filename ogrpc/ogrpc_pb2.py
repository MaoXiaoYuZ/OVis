# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: ogrpc.proto
# Protobuf Python Version: 4.25.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0bogrpc.proto\"\x17\n\x08ORequest\x12\x0b\n\x03pkl\x18\x01 \x01(\x0c\"\x15\n\x06OReply\x12\x0b\n\x03pkl\x18\x01 \x01(\x0c\x32\x45\n\x08OService\x12\x1b\n\x03\x41sk\x12\t.ORequest\x1a\x07.OReply\"\x00\x12\x1c\n\x04Sync\x12\t.ORequest\x1a\x07.OReply\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'ogrpc_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_OREQUEST']._serialized_start=15
  _globals['_OREQUEST']._serialized_end=38
  _globals['_OREPLY']._serialized_start=40
  _globals['_OREPLY']._serialized_end=61
  _globals['_OSERVICE']._serialized_start=63
  _globals['_OSERVICE']._serialized_end=132
# @@protoc_insertion_point(module_scope)
