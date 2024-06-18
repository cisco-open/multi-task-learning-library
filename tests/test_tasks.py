# Copyright 2024 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
import unittest
import torch
from emtl.tasks import Task

class TestTask(unittest.TestCase):
    def test_initialization(self):
        empty_task = Task(
            'example task',
            torch.nn.Linear(1, 1),
            [(0,0), (1,1), (2,2)],
            [(0,0), (1,1), (2,2)],
            lambda x : x,
            lambda x : torch.optim.Adam,
        )
        self.assertIsInstance(empty_task, Task)

if __name__ == '__main__':
    unittest.main()
