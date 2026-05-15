# Copyright 2025-2026 Geoffrey R. Scheller
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
    'DTypes',
    'HWrapNDArray',
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
    """
    .. admonition:: Enumeration of closed NumPy datatypes

        Groups (Unions) of NumPy datatypes closed to NumPy operations.

        - number
        - str\\_
        - bytes
        - datetime64
        - timedelta64
        - bool\\_
        - void
        - object\\_

        While NumPy types are extensively covariant, the NumPy
        C internals are somewhat invariant. NumPy also suffers
        from what I call "Fortran Disease", types get "auto-promoted"
        to compatible "wider types" when necessary. That is fine when
        dealing with operators on mixed types, but NumPy will
        auto-promote with operations on the same type.

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
    stores a read-only copy of the NDArray given to the constructor and
    is hashable.

    """

    __slots__ = '_ndarray', '_type', '_shape', '_hash'

    def __init__(self, ndarray: npt.NDArray[np.generic]) -> None:
        """
        .. admonition:: init

            Base class for a hashable wrapped ndarray.

        """
        self._ndarray = np.array(ndarray, copy=True)
        self._ndarray.setflags(write=False)

        dtype = self._ndarray.dtype.type
        if issubclass(dtype, np.number):
            self._type = DTypes.number
        elif isinstance(dtype, np.str_):
            self._type = DTypes.str_
        elif issubclass(dtype, np.bytes_):
            self._type = DTypes.bytes
        elif issubclass(dtype, np.datetime64):
            self._type = DTypes.datetime64
        elif issubclass(dtype, np.timedelta64):
            self._type = DTypes.timedelta64
        elif issubclass(dtype, np.bool_):
            self._type = DTypes.bool_
        elif issubclass(dtype, np.void):
            self._type = DTypes.void
        elif issubclass(dtype, np.object_):
            self._type = DTypes.object_
        else:
            msg = f"HWrapNDArray: Unknow np.dtype '{dtype}'"
            raise TypeError(msg)

        self._shape = self._ndarray.shape
        self._hash = hash((self._ndarray.tobytes(), hash((self._shape, self._type))))

    def __call__(self) -> npt.NDArray[np.number]:
        """
        .. admonition:: call

            Return a reference to the stored NDArray.

            .. warning::

                For efficiency this method returns a reference to the
                wrapped NDArray, not a copy.

                - Allows for faster slicing and faster operations.
                - Use the copy method if you want a read-write copy.
                - Never make the underlying NDArray writable!!!

            """
        return np.array(self._ndarray)

    def __hash__(self) -> int:
        """
        .. admonition:: hash

            Return hash created during initialization. Hash only
            remains valid if NDArray's flag remains off.

        """
        return self._hash

    def __eq__(self, other: object) -> bool:
        """
        .. admonition:: equality comparison

            :param other: The ``object`` being compared to ``self``.
            :returns: ``True`` if ``object`` is another ``HWrapNDArray``
                      all whose corresponding components are equal to
                      the ones in ``self``.

        """
        if not isinstance(other, type(self)):
            return False
        if self._shape != other._shape or self._type != other._type:
            return False
        return np.array_equal(self._ndarray, other._ndarray)

    def _ndarray_repr(self) -> str:
        ndarray = self._ndarray
        repr_str = repr(ndarray)
        repr_str = repr_str.replace('\n      ', '')
        repr_str = repr_str.replace('  ', ' ')
        repr_str = 'np.' + repr_str[:-1] + ', np.' + repr(ndarray.dtype) + ')'
        return repr_str

    @abstractmethod
    def __repr__(self) -> str: ...

    def __str__(self) -> str:
        """
        .. admonition:: user string

            :returns: String meaningful to an end user.

        """
        user_str = str(self._ndarray)
        user_str = user_str.replace('\n ', '\n    ')
        user_str = '<< ' + user_str + ' >>'
        return user_str

    def copy(self) -> npt.NDArray[np.number]:
        """
        .. admonition:: copy

            Return a copy of the wrapped NDArray.

        """
        return np.array(self._ndarray, copy=True)


class HWrapNDArrayNumber(HWrapNDArray):
    """
    .. admonition:: wrapped numeric ndarray

        Class for a hashable wrapped numeric ndarray.

    """
    def __init__(self, ndarray: npt.NDArray[np.number]) -> None:
        """
        .. admonition:: init

            Wrap NDArray of arbitrary NumPy numeric types.

        """
        super().__init__(ndarray)

    def __repr__(self) -> str:
        """
        .. admonition:: repr string

            :returns: String to reproduce the wrapped numeric ndarray.

        """
        return f'HWrapNDArrayNumber({self._ndarray_repr()})'


class HWrapNDArrayString(HWrapNDArray):
    """
    .. admonition:: wrapped string ndarray

        Class for a hashable wrapped string ndarray.

    """
    def __init__(self, ndarray: npt.NDArray[np.str_]) -> None:
        """
        .. admonition:: init

            Wrap NDArray of Unicode strings.

        """
        super().__init__(ndarray)

    def __repr__(self) -> str:
        """
        .. admonition:: repr string

            :returns: String to reproduce the wrapped string ndarray.

        """
        return f'HWrapNDArrayStr({self._ndarray_repr()})'


class HWrapNDArrayBytes(HWrapNDArray):
    """
    .. admonition:: wrapped byte ndarray

        Class for a hashable wrapped null-terminated
        byte sequence ndarray.

    """
    def __init__(self, ndarray: npt.NDArray[np.bytes_]) -> None:
        """
        .. admonition:: init

            Wrap NDArray of null-terminated byte sequences.

        """
        super().__init__(ndarray)

    def __repr__(self) -> str:
        """
        .. admonition:: repr string

            :returns: String to reproduce the wrapped null-terminated
                      byte sequence ndarray.

        """
        return f'HWrapNDArrayBytes({self._ndarray_repr()})'


class HWrapNDArrayVoid(HWrapNDArray):
    """
    .. admonition:: wrapped void ndarray

        Class for a hashable wrapped arbitrary byte
        sequence ndarray.

    """
    def __init__(self, ndarray: npt.NDArray[np.void]) -> None:
        """
        .. admonition:: init

            Wrap NDArray of arbitrary byte sequences.

        """
        super().__init__(ndarray)

    def __repr__(self) -> str:
        """
        .. admonition:: repr string

            :returns: String to reproduce the wrapped arbitrary
                      byte sequence ndarray.

        """
        return f'HWrapNDArrayVoid({self._ndarray_repr()})'


class HWrapNDArrayObject(HWrapNDArray):
    """
    .. admonition:: wrapped object reference ndarray

        Class for a hashable wrapped object reference ndarray.

    """
    def __init__(self, ndarray: npt.NDArray[np.object_]) -> None:
        """
        .. admonition:: init

            Wrap NDArray of references to arbitrary Python objects.

        """
        super().__init__(ndarray)

    def __repr__(self) -> str:
        """
        .. admonition:: repr string

            :returns: String to reproduce the wrapped object
                      reference ndarray.

        """
        return f'HWrapNDArrayObject({self._ndarray_repr()})'


class HWrapNDArrayDateTime(HWrapNDArray):
    """
    .. admonition:: wrapped datetime ndarray

        Class for a hashable wrapped datatime ndarray.

    """
    def __init__(self, ndarray: npt.NDArray[np.datetime64]) -> None:
        """
        .. admonition:: init

            Wrap NDArray of datetimes.

        """
        super().__init__(ndarray)

    def __repr__(self) -> str:
        """
        .. admonition:: repr string

            :returns: String to reproduce the wrapped datetime ndarray.

        """
        return f'HWrapNDArrayTimeDelta({self._ndarray_repr()})'


class HWrapNDArrayTimeDelta(HWrapNDArray):
    """
    .. admonition:: wrapped timedelta ndarray

        Class for a hashable wrapped timedelta ndarray.

    """
    def __init__(self, ndarray: npt.NDArray[np.timedelta64]) -> None:
        """
        .. admonition:: init

            Wrap NDArray of timedeltas.

        """
        super().__init__(ndarray)

    def __repr__(self) -> str:
        """
        .. admonition:: repr string

            :returns: String to reproduce the wrapped timedelta ndarray.

        """
        return f'HWrapNDArrayTimeDelta({self._ndarray_repr()})'


class HWrapNDArrayBool(HWrapNDArray):
    """
    .. admonition:: wrapped Boolean ndarray

        Class for a hashable wrapped Boolean ndarray.

        .. note::

            NumPy Booleans are actual Booleans, unlike Python bools
            which are subtypes of int.

            - ``*`` uses component-wise Boolean **and**
            - ``+`` uses component-wise Boolean **or**
            - ``@`` matrix multiplication using **and** then **or**

    """

    def __init__(self, ndarray: npt.NDArray[np.bool_]) -> None:
        """
        .. admonition:: init

            Wrap NDArray of Booleans.

        """
        super().__init__(ndarray)

    def __repr__(self) -> str:
        """
        .. admonition:: repr string

            :returns: String to reproduce the wrapped Boolean ndarray.

        """
        return f'HWrapNDArrayBool({self._ndarray_repr()})'
