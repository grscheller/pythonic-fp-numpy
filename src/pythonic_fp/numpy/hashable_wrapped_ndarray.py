# Copyright 2025 Geoffrey R. Scheller
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from enum import auto, Enum
import numpy as np
import numpy.typing as npt

__all__ = [
    'HWrapNDArrayNumber',
    'HWrapNDArrayString',
    'HWrapNDArrayBytes',
    'HWrapNDArrayVoid',
    'HWrapNDArrayObject',
    'HWrapNDArrayDateTime',
    'HWrapNDArrayTimeDelta',
    'HWrapNDArrayBool',
]


class DTypes(Enum):
    """Iterable Blending Enums.

    - **MergeEnum.Concat:** Concatenate first to last
    - **MergeEnum.Merge:** Merge until one is exhausted
    - **MergeEnum.Exhaust:** Merge until all are exhausted

    """

    number = auto()
    str_ = auto()
    bytes = auto()
    void = auto()
    object_ = auto()
    datetime64 = auto()
    timedelta64 = auto()
    bool_ = auto()


class HWrapNDArray(ABC):
    """
    Make NumPy NDArrays hashable.

    Just making an NDArray (np.array) readonly is not enough. This class
    stores a read-only copy of the NDArray given to the constructor. The

    .. warning::

        For efficiency, encapsulation has been deliberately broken. The
        __call__ method returns a reference to the actual underlying
        NDArray, not a copy.

        - Allows for faster slicing. A copy does not have to be made.
        - Use the copy method if you want a read-write copy.

    """

    __slots__ = "_ndarray", "_type", "_shape", "_hash"

    def __init__(self, ndarray: npt.NDArray[np.generic]) -> None:
        self._ndarray = np.array(ndarray, copy=True)
        self._ndarray.setflags(write=False)

        dtype = self._ndarray.dtype.type
        if issubclass(dtype, np.number):
            self._type = DTypes.number
        elif isinstance(dtype, np.str_):
            self._type = DTypes.str_
        elif issubclass(dtype, np.bytes_):
            self._type = DTypes.bytes
        elif issubclass(dtype, np.void):
            self._type = DTypes.void
        elif issubclass(dtype, np.object_):
            self._type = DTypes.object_
        elif issubclass(dtype, np.datetime64):
            self._type = DTypes.datetime64
        elif issubclass(dtype, np.timedelta64):
            self._type = DTypes.timedelta64
        elif issubclass(dtype, np.bool_):
            self._type = DTypes.bool_
        else:
            msg = f"HWrapNDArray: Unknow np.dtype '{dtype}'"
            raise TypeError(msg)

        self._shape = self._ndarray.shape
        self._hash = hash((self._ndarray.tobytes(), hash((self._shape, self._type))))

    @abstractmethod
    def __repr__(self) -> str: ...

    def __call__(self) -> npt.NDArray[np.number]:
        return np.array(self._ndarray)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HWrapNDArray):
            return False
        if self._shape != other._shape or self._type != other._type:
            return False
        return np.array_equal(self._ndarray, other._ndarray)

    def __str__(self) -> str:
        np_array_str = '  ' + str(self._ndarray).replace('\n', '\n  ')
        return f"hwrap<\n{np_array_str}\n>"

    def _repr(self) -> str:
        stripped_repr = " ".join(repr(self._ndarray).split())
        return f"np.{stripped_repr}"

    def copy(self) -> npt.NDArray[np.number]:
        return np.array(self._ndarray, copy=True)


class HWrapNDArrayNumber(HWrapNDArray):
    """Wrap NDArrays of arbitrary NumPy numeric types."""

    def __init__(self, ndarray: npt.NDArray[np.number]) -> None:
        super().__init__(ndarray)

    def __repr__(self) -> str:
        return f"HWrapNDArrayNumber({self._repr()})"


class HWrapNDArrayString(HWrapNDArray):
    """Wrap NDArrays of Unicode strings."""

    def __init__(self, ndarray: npt.NDArray[np.str_]) -> None:
        super().__init__(ndarray)

    def __repr__(self) -> str:
        return f"HWrapNDArrayStr({self._repr()})"


class HWrapNDArrayBytes(HWrapNDArray):
    """Wrap NDArrays of null-terminated byte sequences."""

    def __init__(self, ndarray: npt.NDArray[np.bytes_]) -> None:
        super().__init__(ndarray)

    def __repr__(self) -> str:
        return f"HWrapNDArrayBytes({self._repr()})"


class HWrapNDArrayVoid(HWrapNDArray):
    """Wrap NDArrays of arbitrary byte sequences."""

    def __init__(self, ndarray: npt.NDArray[np.void]) -> None:
        super().__init__(ndarray)

    def __repr__(self) -> str:
        return f"HWrapNDArrayVoid({self._repr()})"


class HWrapNDArrayObject(HWrapNDArray):
    """Wrap NDArrays of references to arbitrary Python objects."""

    def __init__(self, ndarray: npt.NDArray[np.object_]) -> None:
        super().__init__(ndarray)

    def __repr__(self) -> str:
        return f"HWrapNDArrayObject({self._repr()})"


class HWrapNDArrayDateTime(HWrapNDArray):
    """Wrap NDArrays of references to arbitrary Python objects."""

    def __init__(self, ndarray: npt.NDArray[np.datetime64]) -> None:
        super().__init__(ndarray)

    def __repr__(self) -> str:
        return f"HWrapNDArrayTimeDelta({self._repr()})"


class HWrapNDArrayTimeDelta(HWrapNDArray):
    """Wrap NDArrays of references to arbitrary Python objects."""

    def __init__(self, ndarray: npt.NDArray[np.timedelta64]) -> None:
        super().__init__(ndarray)

    def __repr__(self) -> str:
        return f"HWrapNDArrayTimeDelta({self._repr()})"


class HWrapNDArrayBool(HWrapNDArray):
    """Wrap NDArrays of Booleans."""

    def __init__(self, ndarray: npt.NDArray[np.bool_]) -> None:
        super().__init__(ndarray)

    def __repr__(self) -> str:
        return f"HWrapNDArrayBool({self._repr()})"
