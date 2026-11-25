# Copyright 2025 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# ================================================================================================


"""Contains the kinds of tiles and the kinds of z-couplings the package supports."""

from enum import Enum

__all__ = ["TileKind", "ZCoupling"]


class TileKind(Enum):
    LADDER = 0  # Tile has a ladder shape.
    SQUARE = 1  # Tile has a square shape.


class ZCoupling(Enum):
    ZERO_ONE = 0
    ONE_ZERO = 1
    ZERO_ONE_ONE_ZERO = 2
    EITHER = 3
