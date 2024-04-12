from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class ORequest(_message.Message):
    __slots__ = ("pkl",)
    PKL_FIELD_NUMBER: _ClassVar[int]
    pkl: bytes
    def __init__(self, pkl: _Optional[bytes] = ...) -> None: ...

class OReply(_message.Message):
    __slots__ = ("pkl",)
    PKL_FIELD_NUMBER: _ClassVar[int]
    pkl: bytes
    def __init__(self, pkl: _Optional[bytes] = ...) -> None: ...
