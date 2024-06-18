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
import torch
import collections
from typing import Callable
from functools import partial


def aggregate_metrics(metrics_dicts: list[dict[str, float]]) -> dict[str, float]:
    '''
    Average a list of dictionaries by corresponding keys.
    An example can show this best:
    ```
    dictA = {'a': 10, 'b': 20}
    dictB = {'a': 0,  'b': 20, 'c': 6}
    dictc = aggregate_metrics([dictA, dictB])
    ```
    In this case, we will find that `dictC` has value `{'a': 5,  'b': 20, 'c': 6}`.

    Args:
        metrics_dicts (list[dict[str, float]]): list of dictionaries of key-value pairs.

    Returns:
        dict[str, float]: a single dictionary with all the keys and the average value of each.
    '''
    sums = collections.defaultdict(lambda : 0)
    cnts = collections.defaultdict(lambda : 0)

    for dictionary in metrics_dicts:
        for key in dictionary:
            sums[key] += dictionary[key]
            cnts[key] += 1
    
    aggregated_dictionary = {key : sums[key] / cnts[key] for key in sums}

    return aggregated_dictionary


def infinite_data_batch_producer(dataloader: torch.utils.data.DataLoader
                               ) -> tuple[tuple[torch.tensor, torch.tensor], bool]:
    '''
    Create a generator of batches of data from a DataLoader.
    This method returns an infinite (cyclic) iterator over the given DataLoader. Each item is a
    tuple, where the first item is a batch of data (a pair of inputs and targets), and the
    second is a boolean value that is true every time a cycle over the underlying dataset is
    completed. If you want to iterate through the dataset once, simply loops until the second
    item in the tuple becomes true.

    Args:
        dataloader (torch.utils.data.DataLoader): a PyTorch DataLoader.

    Returns:
        tuple[tuple[torch.tensor, torch.tensor], bool]: pairs of ((inputs, targets), is_last_batch).

    Yields:
        Iterator[tuple[tuple[torch.tensor, torch.tensor], bool]]: infinite (cyclic) dataloader.
    '''
    n = len(dataloader)

    while True:
        batch_num = 0

        for batch in dataloader:
            batch_num += 1
            yield batch, batch_num == n


def config_file_section_to_dict(config_parser, section: str):
    dictionary = {}

    if section not in config_parser:
        return {}

    for key, value in config_parser[section].items():
        # check if integer
        try:
            dictionary[key] = int(value)
            continue
        except: pass

        # check if float
        try:
            dictionary[key] = float(value)
            continue
        except: pass
        
        # check if bool
        if value.lower() in ['true', 'false']:
            dictionary[key] = value.lower() == 'true'
            continue
        
        # must be string then
        dictionary[key] = value
    
    return dictionary


class Logger():
    log: Callable
    step: int

    def __init__(self, use_mlflow: bool = False, out_file: str = None) -> None:
        self.step = 0
        
        if use_mlflow:
            self.log = self.log_metrics_with_mlflow
        
        elif out_file:
            self.log = partial(self.log_metrics_to_file, filename=out_file)
        
        else:
            self.log = lambda *args, **kwargs: None
    
    def log_metrics_with_mlflow(self, metrics_dictionary: dict[str, float]) -> None:
        '''
        Wrapper method to isolate the MLFlow dependency from global scope.

        Args:
            metrics (dict[str, float]): dictionary of named metric values to log.
        '''
        import mlflow
        mlflow.log_metrics(metrics_dictionary, step=self.step)
    
    def log_metrics_to_file(self, metrics_dictionary: dict[str, float], filename: str) -> None:
        with open(filename, '+a') as f:
            f.write(f'Step {self.step}')
            f.write(metrics_dictionary)
    
    def increase_step(self) -> None:
        self.step += 1
