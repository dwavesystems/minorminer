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


from unittest import TestCase

from minorminer._lattice_utils.auxiliary_coordinates import *


class TestUWKJZ(TestCase):
    def test_uwkj(self):
        nodes = [UWKJZ(u=0, w=2, k=3, j=1, z=0), UWKJZ(u=1, w=10, k=0, j=0, z=1)]
        for node in nodes:
            external_path = node.uwkj
            for field in ("u", "w", "k", "j"):
                self.assertEqual(getattr(node, field), getattr(external_path, field))

    def test_uwj(self):
        nodes = [UWKJZ(u=0, w=2, k=3, j=1, z=0), UWKJZ(u=1, w=10, k=0, j=0, z=1)]
        for node in nodes:
            quotient_external_path = node.uwj
            for field in ("u", "w", "j"):
                self.assertEqual(getattr(node, field), getattr(quotient_external_path, field))


class TestUWJZ(TestCase):
    def test_uwj(self):
        ext_paths = [UWKJ(u=1, w=2, k=3, j=1), UWKJ(u=1, w=10, k=20, j=0), UWKJ(u=0, w=0, k=0, j=0)]
        for ext_path in ext_paths:
            quotient_external_path = ext_path.uwj
            for field in ("u", "w", "j"):
                self.assertEqual(getattr(ext_path, field), getattr(quotient_external_path, field))
