import pickle
from abc import ABC
from dataclasses import dataclass

@dataclass 
class AbstractDataclass(ABC): 
    """ An abstract dataclass that prohibits direct instantiation. """
    # Reference: https://stackoverflow.com/questions/60590442/abstract-dataclass-without-abstract-methods-in-python-prohibit-instantiation
    def __new__(cls, *args, **kwargs): 
        if cls == AbstractDataclass or cls.__bases__[0] == AbstractDataclass: 
            raise TypeError("Cannot instantiate abstract class.") 
        return super().__new__(cls)
    
class ByteSerializable(ABC):
    """ An abstract class that can be serialized to bytes. """
    
    def to_bytes(self) -> bytes:
        return pickle.dumps(self)
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'ByteSerializable':
        return pickle.loads(data)