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
from abc import ABC, abstractmethod
from collections.abc import Callable
from configparser import ConfigParser

import torch
from tqdm import tqdm

from emtl.utils import aggregate_metrics, infinite_data_batch_producer, config_file_section_to_dict


class Task(ABC):
    name: str
    trainloader: torch.utils.data.DataLoader
    testloader: torch.utils.data.DataLoader
    optimizer_fn: Callable[..., torch.optim.Optimizer]
    config: str
    device: str

    def __init__(self,
                name: str, 
                trainset: torch.utils.data.Dataset,
                testset: torch.utils.data.Dataset,
                optimizer_fn: Callable[..., torch.optim.Optimizer],
                config: str = '',
                dataloader_params: dict[str, any] = {},
                optimizer_params: dict[str, any] = {},
                ) -> None:
        '''
        Create a new basic Task.
        A task is defined by its datasets (train and test), the optimizer functions and learning 
        rate schedulers, and one or several specialized heads.

        The optimizer, and optionally a learning rate scheduler, serve to update a model's parameters
        and must not be passed as instantiated objects, but rather functions. As tasks are logically
        separated from backbones (a task only sees its specialized head), optimiers are created at
        runtime by a Trainer object.

        Data Loader params are those that define how much data to read at once, and how. These are
        passed as-is to the dataloaders that wrap the train and test datasets. One can specify things
        like batch size, number of workers, etc. 
        See https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader for more info.

        A configuration file (.INI format) may be specified too, to include dataloader, optimizer, 
        scheduler, and global parameters. When the same parameter is specified via config file and
        via method parameters, the latter take priority. For examples on how to write a config file,
        see `./configs/tasks/MNIST.ini`.

        Args:
            name (str): a friendly name for the task (can be the dataset name too)
            trainset (torch.utils.data.Dataset): dataset of train samples
            testset (torch.utils.data.Dataset): dataset of test samples
            optimizer_fn (Callable[..., torch.optim.Optimizer]): function that *produces* an optimizer
            config (str, optional): path to a configuration file, in alternative to passing 
                parameters. Defaults to ''.
            dataloader_params (dict[str, any], optional): paramters for the dataloader. Defaults to {}.
            optimizer_params (dict[str, any], optional): paramters for the optimizer. Defaults to {}.
        '''
        self.name = name
        self.optimizer_fn = optimizer_fn

        self.parse_configurations(config, dataloader_params, optimizer_params)
        self.instantiate_dataloaders(trainset, testset)
    

    def parse_configurations(self, config: str, dataloader_params: dict[str, any], 
                             optimizer_params: dict[str, any]
                             ) -> None:
        '''
        Internal helper method to process a configuration file and method parameters.
        This method merges the (optional) configuration within the config file provided (of course,
        if any) with the method arguments passed in the function call. When parameters are duplicate,
        those passed in the method call take precedence.

        Args:
            config (str): path to a config.ini file
            dataloader_params (dict[str, any]): parameters to pass to the dataloader.
            optimizer_params (dict[str, any]): parameters to pass to the optimizer.
        '''
        parser = ConfigParser()
        parser.read(config)

        # default, possibly empty dictionaries
        self.dataloader_params = config_file_section_to_dict(parser, 'dataloader')
        self.optimizer_params  = config_file_section_to_dict(parser, 'optimizer')
        self.device = parser.get('global', 'device', fallback='cpu')

        # update values with passed arguments (as they take precedence)
        self.dataloader_params.update(dataloader_params)
        self.optimizer_params.update(optimizer_params)

        # if memory is pinned, then also enable non blocking RAM -> GPU-RAM tranfers
        self.non_blocking_ops = 'pin_memory' in self.dataloader_params and self.dataloader_params['pin_memory']


    def instantiate_dataloaders(self, trainset: torch.utils.data.Dataset, 
                                testset: torch.utils.data.Dataset) -> None:
        '''
        Helper method to create the DataLoaders. This method will also create the train data producer.

        Args:
            trainset (torch.utils.data.Dataset): train dataset.
            testset (torch.utils.data.Dataset): test dataset.
        '''
        self.trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, **self.dataloader_params)
        self.testloader = torch.utils.data.DataLoader(testset, shuffle=False, **self.dataloader_params)
        self.trainproducer = infinite_data_batch_producer(self.trainloader)
        

    @abstractmethod
    def train_step(self, backbone: torch.nn.Module) -> bool:
        raise NotImplementedError()
    

    @abstractmethod
    def eval(self, backbone: torch.nn.Module, set: str = 'test') -> dict[str, float]:
        raise NotImplementedError()


    @abstractmethod
    def reset_optimizers(self):
        raise NotImplementedError()



class SimpleTask(Task):
    head: torch.nn.Module
    criterion: torch.nn.modules.loss._Loss
    optimizer: torch.optim.Optimizer
    metric_fns: dict[str, Callable[..., float]]

    def __init__(self,
                name: str, 
                head: torch.nn.Module, 
                criterion: torch.nn.modules.loss._Loss,
                metric_fns: dict[str, Callable[..., float]] = {},
                *args, **kwargs
                ) -> None:
        '''
        Create a new Task with a single specialized head.
        At minimum, this class requires a `name`, a `trainset` and a `testset`, and an `optimizer` function.
        Refer to the documentation of the parent class`Task` for more information about these and
        other optional parameters.

        This object corresponds to a single task with a single dataset. It therefore requires a `head`
        model, a `criterion` to compute the loss, and it supports optional evaluation metrics.

        `metric_fns` is a named dictionary of functions that provide additional info
        into the accuracy of the backbone/head for a task, beyond just the loss/criterion. For
        example, one may list 'accuracy' in a multiclass classification task, and RMSE loss too. For
        these, one would pass a dictionary as follows:
        ```
        metric_fns = {
            "accuracy": lambda pred, true : (pred.argmax(1) == true).float().item(),
            "rmse": lambda pred, true : torch.sqrt(nn.MSELoss()(pred, true))
        }
        ```

        Args:
            name (str): a friendly name for the task (can be the dataset name too)
            head (torch.nn.Module): model to post-process the encoder's embeddings (likely a MLP)
            criterion (torch.nn.modules.loss._Loss): function to calculate the loss
            metric_fns (dict[str, Callable[..., float]], optional): see description above. Defaults to {}.
        '''
        super(SimpleTask, self).__init__(name, *args, **kwargs)

        self.head = head.to(self.device, non_blocking=self.non_blocking_ops)
        self.metric_fns = metric_fns
        self.criterion = criterion
        self.reset_optimizers()

    
    def reset_optimizers(self):
        self.head_optimizer = self.optimizer_fn(self.head.parameters(), **self.optimizer_params)
        

    def train_step(self, backbone: torch.nn.Module) -> tuple[bool, float]:
        '''
        Perform one forward and backward pass.
        This function consumes a single batch of data from the `trainproducer`, feeds it to the
        backbone and the head, calculates the loss, and does backpropagation. 

        Args:
            backbone (torch.nn.Module): backbone model to produce embeddings.
            freeze_backbone (bool): whether to stop the backbone from training (True) or update it (False).

        Returns:
            bool: whether the batch of data just processed was at the end of the train dataset.
            float: the loss of the batch
        '''
        (inputs, target), is_last_batch = next(self.trainproducer)
        inputs, target = inputs.to(self.device, non_blocking=self.non_blocking_ops), target.to(self.device, non_blocking=self.non_blocking_ops)

        # clear the gradients
        self.head_optimizer.zero_grad()

        # compute features, output and loss
        features = backbone(inputs)
        pred = self.head(features, original_shape = inputs.shape[-2:])
        loss = self.criterion(pred, target)
        loss.backward()
        self.head_optimizer.step()
        
        return is_last_batch, loss
    

    def eval(self, backbone: torch.nn.Module, set: str = 'test') -> dict[str, float]:
        '''
        Evaluate the performance of the head and backbone on a whole dataset.

        Args:
            backbone (torch.nn.Module): the model to compute embeddings.
            set (str, optional): the dataset to evaluate on. Can be `train` or `test`. Defaults to 'test'.

        Returns:
            dict[str, float]: named dictionary of the metrics evaluated, and the loss
        '''
        dataloader = self.trainloader if set == 'train' else self.testloader
        metrics_list = []

        with torch.no_grad():
            for (inputs, target) in tqdm(dataloader, desc=self.name+' eval'):
                inputs, target = inputs.to(self.device, non_blocking=self.non_blocking_ops), target.to(self.device, non_blocking=self.non_blocking_ops)

                # forward pass
                features = backbone(inputs)
                pred = self.head(features, original_shape = inputs.shape[-2:])
                loss = self.criterion(pred, target)

                # evaluate metrics
                metrics_list.append({
                    **{mname:  self.metric_fns[mname](pred, target) for mname in self.metric_fns},
                    'loss': loss.item()
                })
                
        return aggregate_metrics(metrics_list)



class MultiHeadedDatasetTask(Task):
    tasks_specs: list[tuple[
        str,                                # subtask name
        torch.nn.Module,                    # head model
        torch.nn.modules.loss._Loss,        # criterion
        dict[str, Callable[..., float]]     # metric_fns
    ]]
    optimizers: list[torch.optim.Optimizer]

    def __init__(self,
                name: str, 
                tasks_specs: list[tuple[
                    str, 
                    torch.nn.Module, 
                    torch.nn.modules.loss._Loss, 
                    dict[str, Callable[..., float]]]],
                *args, **kwargs) -> None:
        '''
        Create a new Task with multiple specialized heads.
        At minimum, this class requires a `name`, a `trainset` and a `testset`, and an `optimizer` function.
        Refer to the documentation of the parent class`Task` for more information about these and
        other optional parameters. Both the trainset and the testste must return a single input data
        point (an image or a batch), and one target per each given head in the `tasks_specs`.

        This object corresponds to a collection of tasks that are based on the same dataset. For example,
        the Pascal VOC dataset can be used for both object detection and semantic segmentation. The
        advantage of using this object over creating several `SimpleTask` instances with the same
        datasets is that the *encoding* part (processing images with the backbone) is done only once
        per image for a set of tasks; then, each task's head is fed the generated embedding and optimized.
        If multiple instances of single-headed tasks are used, the backbone would have to process every
        train/test sample once per task, which is slower. From a learning perspective, the two approaches
        are equivalent: this object accumulates gradients on the backbone for each task, and then updates
        its values once at the end, with the cumulative updates from each task.

        The `tasks_specs` required parameter describes the tasks to attach to the dataset. This is
        therefore a list of specifications. Each specification is a quadruplet, with the following
        components (note that here they are named for reference, but the parameter is a tuple, not a dict):
        - `name` is the name of the specific task. For instance, one may have Pascal VOC be the name
            of the `MultiHeadedDatasetTask` object, and `Semantic Segmentation` be this task's name;
        - `head` is the model that processes the embeddings produced by the backbone for the task;
        - `criterion` is the function to compute the loss
        - `metric_fns` is a named dictionary of evaluation functions beyond the loss. See `SimpleTask`
            for a more detailed explanation and examples.

        Args:
            name (str): a friendly name for the task (can be the dataset name too)
            tasks_specs (list[
                tuple[ 
                    str, 
                    torch.nn.Module, 
                    torch.nn.modules.loss._Loss, 
                    dict[str, Callable[..., float]]
                ]]): list of heads' specifications as described above
        '''
        super(MultiHeadedDatasetTask, self).__init__(name, *args, **kwargs)
        self.tasks_specs = []
        
        # change the list of tuples to list of dictinaries to more easily access its members
        for t_name, head, criterion, metric_fns in tasks_specs:
            head = head.to(self.device, non_blocking=self.non_blocking_ops)
            optimizer = self.optimizer_fn(head.parameters(), **self.optimizer_params)

            self.tasks_specs.append({
                'name': t_name, 
                'head': head,
                'criterion': criterion, 
                'metrics': metric_fns,
                'optimizer': optimizer,
            })

    
    def reset_optimizers(self):
        for spec in self.tasks_specs:
            spec['optimizer'] = self.optimizer_fn(spec['head'].parameters(), **self.optimizer_params)
        

    def train_step(self, backbone: torch.nn.Module) -> tuple[bool, float]:
        '''
        Perform one forward and backward pass.
        This function consumes a single batch of data from the `trainproducer` and feeds it to the
        backbone once to produce the embeddings.
        Then, for every head in the `task_specs`, a target is taken from the batch of data, and the
        embeddings are passed through the head, to then compare with the targets and compute the loss.
        The loss is backpropagated through the head and the backbone both. Then, the head's weights
        are updated. Once all heads have been updated, the backbone is updated with the accumulated
        gradients information.

        Args:
            backbone (torch.nn.Module): backbone model to produce embeddings.
            freeze_backbone (bool): whether to stop the backbone from training (True) or update it (False).

        Returns:
            bool: whether the batch of data just processed was at the end of the train dataset.
            float: cumulative loss for the task
        '''
        (inputs, targets), is_last_batch = next(self.trainproducer)
        inputs = inputs.to(self.device, non_blocking=self.non_blocking_ops)
        features = backbone(inputs)
        overall_loss = 0

        for target, task in zip(targets, self.tasks_specs):
            target = target.to(self.device, non_blocking=self.non_blocking_ops)

            # forward prop
            task['head'].zero_grad()
            pred = task['head'](features, original_shape=inputs.shape[-2:])

            # compute and accumulate loss
            loss = task['criterion'](pred, target)
            overall_loss = overall_loss + loss

        # compute all gradients
        overall_loss.backward()

        # update the heads' weights
        for task in self.tasks_specs:
            task['optimizer'].step()
        
        return is_last_batch, overall_loss
    

    def eval(self, backbone: torch.nn.Module, set: str = 'test') -> dict[str, float]:
        '''
        Evaluate the performance of the backbone and all the heads on a whole dataset.
        This method produces a named dictionary of metrics, based on the supplied `metric_fns`
        dictionaries in each `task_spec`. The produced dictionary contains a key for every task-metric
        pair, for instance a `task_specs` with a task named `bbox` and another task named `features`,
        the latter with a `metric_fns` dictionary with a metric called `precision`, will produce a 
        dictionary with keys `['bbox loss', 'features precision', 'features loss']`. 

        Args:
            backbone (torch.nn.Module): the model to compute embeddings.
            set (str, optional): the dataset to evaluate on. Can be `train` or `test`. Defaults to 'test'.

        Returns:
            dict[str, float]: named dictionary of the metrics evaluated, and the loss, for each task.
        '''
        dataloader = self.trainloader if set == 'train' else self.testloader
        metrics_list = []

        with torch.no_grad():
            for (inputs, targets) in tqdm(dataloader, desc=self.name+' eval'):
                features = backbone(inputs.to(self.device, non_blocking=self.non_blocking_ops))

                for target, task in zip(targets, self.tasks_specs):
                    target = target.to(self.device, non_blocking=self.non_blocking_ops)

                    # forward pass
                    pred = task['head'](features, original_shape = inputs.shape[-2:])
                    loss = task['criterion'](pred, target)

                    # evaluate metrics
                    metric_fns = task['metrics']
                    tname = task['name']

                    metrics_list.append({
                        **{f'{tname} {mname}':  metric_fns[mname](pred, target) for mname in metric_fns},
                        f'{tname} loss': loss.item()
                    })
        
        return aggregate_metrics(metrics_list)

