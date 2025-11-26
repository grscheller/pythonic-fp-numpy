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

import numpy as np
import numpy.typing as npt


class HWrapNDArray:
    """
    Make NumPy NDArrays hashable.

    Just making an NDArray (np.array) readonly is not enough. This class
    stores a read-only copy of the NDArray given to the constructor. The
    __call__ method will return a read-write copy of the stored NDArray.

    .. warning::

        API will change to allow either __call__ to return a copy or for
        efficiency a actual reference to the stored object.

    """

    __slots__ = "_ndarray", "_dtype", "_shape", "_hash"

    def __init__(self, ndarray: npt.NDArray[np.generic]) -> None:
        self._ndarray = np.array(ndarray, copy=True)
        self._ndarray.setflags(write=False)
        self._dtype = str(self._ndarray.dtype)
        self._shape = self._ndarray.shape
        self._hash = hash((self._ndarray.tobytes(), hash((self._shape, self._dtype))))

    def __repr__(self) -> str:
        return f"HWrapNDArray({self._repr()})"

    def __call__(self) -> npt.NDArray[np.number]:
        return np.array(self._ndarray, copy=True)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HWrapNDArray):
            return False
        if self._shape != other._shape or self._dtype != other._dtype:
            return False
        return np.array_equal(self._ndarray, other._ndarray)

    def _repr(self) -> str:
        stripped_repr = " ".join(repr(self._ndarray).split())
        return f"np.{stripped_repr}"

    def __str__(self) -> str:
        return f"hwap<\n{str(self._ndarray)}\n>"


class HWrapNDArrayBool(HWrapNDArray):
    """Wrap NDArrays of Booleans."""

    def __init__(self, ndarray: npt.NDArray[np.bool_]) -> None:
        super().__init__(ndarray)

    def __repr__(self) -> str:
        return f"HWrapNDArrayBool({self._repr()})"


class HWrapNDArrayStr(HWrapNDArray):
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


class HWrapNDArrayNumber(HWrapNDArray):
    """Wrap NDArrays of arbitrary NumPy numeric types."""

    def __init__(self, ndarray: npt.NDArray[np.number]) -> None:
        super().__init__(ndarray)

    def __repr__(self) -> str:
        return f"HWrapNDArrayNumber({self._repr()})"


# TODO: Breakout all the numeric types
