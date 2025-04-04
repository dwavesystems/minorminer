# Copyright 2025 D-Wave
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


from __future__ import annotations

from collections import namedtuple

zephyr_fields = ["u", "w", "k", "j", "z"]
ZephyrCoord = namedtuple("ZephyrCoord", zephyr_fields, defaults=(None,) * len(zephyr_fields))
cartesian_fields = ["x", "y", "k"]
CartesianCoord = namedtuple(
    "CartesianCoord", cartesian_fields, defaults=(None,) * len(cartesian_fields)
)


def cartesian_to_zephyr(ccoord: CartesianCoord) -> ZephyrCoord:
    """Converts a CartesianCoord to its corresponding ZephyrCoord.
    Note: It assumes the given CartesianCoord is valid.

    Args:
        ccoord (CartesianCoord): The coodinate in Cartesian system to be converted.

    Returns:
        ZephyrCoord: The coordinate of the ccoord in Zephyr system.
    """
    x, y, k = ccoord
    if x % 2 == 0:
        u: int = 0
        w: int = x // 2
        j: int = ((y - 1) % 4) // 2
        z: int = y // 4
    else:
        u: int = 1
        w: int = y // 2
        j: int = ((x - 1) % 4) // 2
        z: int = x // 4
    return ZephyrCoord(u=u, w=w, k=k, j=j, z=z)


def zephyr_to_cartesian(zcoord: ZephyrCoord) -> CartesianCoord:
    """Converts a ZephyrCoord to its corresponding CartesianCoord.
    Note: It assumes the given ZephyrCoord is valid.

    Args:
        zcoord (ZephyrCoord): The coodinate in Zephyr system to be converted.

    Returns:
        CartesianCoord: The coordinate of the ccoord in Cartesian system.
    """
    u, w, k, j, z = zcoord
    if u == 0:
        x = 2 * w
        y = 4 * z + 2 * j + 1
    else:
        x = 4 * z + 2 * j + 1
        y = 2 * w
    return CartesianCoord(x=x, y=y, k=k)
