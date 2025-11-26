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
from typing import Any
from boring_math.abstract_algebra.algebras.semigroup import Semigroup

## Infrastructure setup


class HashableNDArrayWrapper:
    __slots__ = '_ndarray', '_array', '_hash', '_shape'

    def __init__(self, ndarray: npt.NDArray[np.int32]) -> None:
        self._ndarray = np.array(ndarray, copy=True)
        self._ndarray.setflags(write=False)
        self._hash = hash(
            (
                self._ndarray.tobytes(),
                hash((self._ndarray.shape, self._ndarray.dtype)),
            )
        )

    def __call__(self) -> npt.NDArray[Any]:
        return np.array(self._ndarray, copy=True)

    def __str__(self) -> str:
        return f'hw<{str(self._ndarray)}>'

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HashableNDArrayWrapper):
            return NotImplemented
        if (
            self._ndarray.shape != other._ndarray.shape
            or self._ndarray.dtype != other._ndarray.dtype
        ):
            return False
        return np.array_equal(self._ndarray, other._ndarray)


## Implementation

type I32_2x2 = HashableNDArrayWrapper[npt.NDArray[np.int32]]


def matrix_mult(left: I32_2x2, right: I32_2x2) -> I32_2x2:
    return HashableNDArrayWrapper(left() @ right())


m2x2 = Semigroup[I32_2x2](mult=matrix_mult)

np_eye = HashableNDArrayWrapper(np.eye(2, dtype=np.int32))
np_zero = HashableNDArrayWrapper(np.zeros((2, 2), dtype=np.int32))
np_A = HashableNDArrayWrapper(np.array([[5, -1], [0, 2]], dtype=np.int32))
np_B = HashableNDArrayWrapper(np.array([[2, -1], [-1, 2]], dtype=np.int32))
np_C = HashableNDArrayWrapper(np.array([[1, 1], [1, 1]], dtype=np.int32))
np_D = HashableNDArrayWrapper(np.array([[0, 1], [1, 0]], dtype=np.int32))
np_E = HashableNDArrayWrapper(np.array([[11, -7], [-2, 4]], dtype=np.int32))


Eye = m2x2(np_eye)
Zero = m2x2(np_zero)
A = m2x2(np_A)
B = m2x2(np_B)
C = m2x2(np_C)
D = m2x2(np_D)
E = m2x2(np_E)


class Test_bool3:
    def test_equality(self) -> None:
        assert Eye * Eye == Eye
        assert Eye * A == A
        assert B * Eye == B
        assert E * Zero == Zero
        assert Zero * E == Zero
        assert (A * B) * C == A * (B * C)
        assert D * D == Eye
        assert A * B == E

    def test_identity(self) -> None:
        assert Eye * Eye is Eye
        assert Eye * A is A
        assert B * Eye is B
        assert E * Zero is Zero
        assert Zero * E is Zero
        assert (A * B) * C is A * (B * C)
        assert D * D is Eye
        assert A * B is E

    def test_create(self) -> None:
        np_see = HashableNDArrayWrapper(np.array([[1, 1], [1, 1]], dtype=np.int32))
        See = m2x2(np_see)
        assert See == C
        assert See is C

    def test_pow(self) -> None:
        Eye**5 == Eye
        Eye**5 is Eye

    def test_annotated(self) -> None:
        assert str(A()) == 'hw<[[ 5 -1]\n [ 0  2]]>'
