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
from collections import deque
from typing import Callable, Union

import torch
from tqdm import trange, tqdm

from emtl.tasks import Task
from emtl.metrics import normalized_entropy, average_pearson_product_moment_correlation_coefficient
from emtl.utils import Logger


class DummyLRScheduler:
    def __init__(self, *args, **kwargs) -> None:
        pass
    
    def step(self, *args, **kwargs) -> None:
        pass


class TrainingAlgorithm(ABC):
    device: str
    logger: Logger
    epochs: int
    min_adk: float = 1

    def __init__(self,
                epochs: int,
                optimizer_fn: Callable[..., torch.optim.Optimizer],
                optimizer_params: dict[str, any],
                scheduler_fn: Callable[..., torch.optim.lr_scheduler._LRScheduler] = DummyLRScheduler,
                scheduler_params: dict[str, any] = {},
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu', 
                rmc_lookback: int = 5,
                logger: Logger = Logger(),
                freeze_backbone: bool = False
                ) -> None:
        '''
        Abstract class to define the common behavior of a training algorithm.

        Args:
            device (str, optional): device to move data to. Defaults to 'cpu'.
            rmc_lookback (int, optional): epochs window to use to calculate RMC. Defaults to 5.
        '''
        self.epochs = epochs
        self.optimizer_fn = optimizer_fn
        self.scheduler_fn = scheduler_fn
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params

        self.device = device
        self.rmc_lookback = rmc_lookback
        self.logger = logger
        self.freeze_backbone = freeze_backbone

        self.losses = {}
        self.rmcs = {}
        self.non_blocking_ops = device.lower() != 'cpu'
        self.optimizer_is_configured = False
    

    def configure_optimizer(self, backbone: torch.nn.Module) -> None:
        self.optimizer_is_configured = True
        self.optimizer = self.optimizer_fn(backbone.parameters(), **self.optimizer_params)
        self.scheduler = self.scheduler_fn(self.optimizer, **self.scheduler_params)
    

    def reset_optimizers(self, backbone: torch.nn.Module) -> None:
        self.configure_optimizer(backbone)

    
    def indipendent_capacity_evaluation(self, backbone: torch.nn.Module, min_t: float = 0.001) -> None:
        '''
        Evaluate the capacity of the model without any dependencies.
        This method evaluates the capacity based on what percentage of the model's weights are
        weakly activated: the more weights are close to 0, the more capacity the model has left.
        The percentage is that of weights less than one standard deviation from the mean of that layer.
        Calculations are made layer-wise.

        Args:
            backbone (torch.nn.Module): model that generates features.
            min_t (float): minimum threshold to consider; a smaller one won't be picked.
        '''
        backbone.eval()

        # we keep two counts: the # of small weights, and the # of all weights
        # for each layer, we find the mean and standard deviation, and count the weigts < 1 standard dev
        ratios = []

        for _, parameters in backbone.named_parameters():
            absolute_values = abs(parameters).detach().flatten()
            mean = torch.mean(absolute_values)
            std = torch.std(absolute_values)
            threshold = max(mean - std, min_t)

            small = (absolute_values <= threshold).sum()
            total = len(absolute_values)
            ratios.append((small / total).item())

        self.logger.log({'I-Capacity': sum(ratios) / len(ratios)})


    def data_dependent_capacity_evaluation(self, 
            backbone: torch.nn.Module, 
            dataloaders: Union[list[torch.utils.data.DataLoader], torch.utils.data.DataLoader]
            ) -> None:
        '''
        Evaluate the capacity of a model given a representative sample of data.
        Given a single or multiple dataloaders, it runs through each of them and stacks the predictions 
        of the model. Then, given those predictions, or features, it computes several capacity metrics.
        This method may take some time to run.

        Args:
            backbone (torch.nn.Module): model that generates features.
            dataloaders (Union[list[torch.utils.data.DataLoader], torch.utils.data.DataLoader]): 
                a single dataloader or a list of dataloaders.
        '''
        backbone.eval()
        
        # make sure you have either one dataset or a list of datasets
        if not isinstance(dataloaders, list):
            dataloaders = [dataloaders]
        outputs = []
        
        # now produce and save features for all passed data
        with torch.no_grad():
            for dataloader in dataloaders:
                for inputs, _ in dataloader:
                    features = backbone(inputs.to(self.device, non_blocking=self.non_blocking_ops))
                    outputs.append(features.cpu())
        
        # create a [d x n] tensor (d = # of features, n = sum of dataloaders lengths)
        outputs = torch.vstack(outputs).T

        # compute and log capacity metrics
        entropy = normalized_entropy(outputs, nbins=100)
        pearson = average_pearson_product_moment_correlation_coefficient(outputs)

        self.logger.log({
            'Entropy': entropy,
            'Correlation': pearson,
            'DD-Capacity': ((pearson - entropy + 1) / 2)
        })


    def algo_dependent_capacity_evaluation(self, min_t: float = 0.01) -> None:
        '''
        Evaluate the capacity of a model given its past performance while training with this algo.
        This method takes no arguments because it's using what the algorithm has already computed
        while training the backbone/heads. 
        This method simply gives the percentage of tasks for which the RMC is above 0. 
        Could be improved in the future to consider a threshold.
        '''
        if len(self.rmcs) == 0:
            self.logger.log({'AD-Capacity': 1})
            return
        
        positive_count = len([rmc for rmc in self.rmcs.values() if rmc > min_t])
        total = len(self.rmcs)
        capacity = positive_count / total
        
        # make it no less than min existing one
        capacity = min(capacity, self.min_adk)
        self.min_adk = capacity

        self.logger.log({'AD-Capacity': capacity})


    def full_task_eval(self, backbone: torch.nn.Module, task: Task) -> None:
        '''
        Perform a full evaluation of the backbone model.
        This method takes in either a SimpleTask or a MultiHeadedDatasetTask, and for each task
        within, performs an evaluation pass. Evaluation is done on both the trainset and the testset.

        Args:
            backbone (torch.nn.Module): backbone model to evaluate.
            task (Task): task to evaluate the backbone for.
        '''
        backbone.eval()
        
        # compute dictionaries of <metric_name, value> pairs 
        train_metrics = task.eval(backbone, set='train')
        test_metrics  = task.eval(backbone, set='test')
        rmcs = self.compute_all_rmcs(task.name, train_metrics)

        # rename the dictionary keys to include the main task name (dataset) and split
        train_metrics = {f'{task.name} train {mname}' : v for mname, v in train_metrics.items()}
        test_metrics  = {f'{task.name} test {mname}'  : v for mname, v in test_metrics.items()}
        rmcs          = {f'{mname} RMC' : v for mname, v in rmcs.items()}

        # log all metrics
        self.logger.log(train_metrics)
        self.logger.log(test_metrics)
        self.logger.log(rmcs)


    def compute_all_rmcs(self, dataset_name: str, metrics_dict: dict[str, float]) -> dict[str, float]:
        '''
        Compute Residual Model Capacity metrics for all losses in dictionary.

        Args:
            dataset_name (str): name of the main task / dataset.
            metrics_dict (dict[str, float]): dictionary that should contain some '... loss' metrics.

        Returns:
            dict[str, float]: a named dictionary of task losses (named by dataset + task name)
        '''
        rmcs_dict = {}

        for metric_name in metrics_dict:
            # name could be like "BBox loss" or just "loss"
            metric_name_components = metric_name.split(' ')
            mname = metric_name_components[-1]
            task_name = (' ' + metric_name_components[0]) if (len(metric_name_components) == 2) else ''

            # proceed forward only if the metric is indeed a loss
            if mname == 'loss':
                loss_value = metrics_dict[metric_name]
                log_name = f'{dataset_name}{task_name}'

                rmc = self.compute_individual_rmc(log_name, loss_value)

                rmcs_dict[log_name] = rmc
                self.rmcs[log_name] = rmc

        return rmcs_dict


    def compute_individual_rmc(self, log_name: str, loss: float) -> float:
        '''
        Method to compute the Residual Model Capacity with respect to a single task.
        This method requires the task name to search in its dictionary of saved losses, and the
        latest loss for that task (which gets saved for the next computation).

        Args:
            log_name (str): key of the dictionary.
            loss (float): latest loss value for the key.

        Returns:
            float: newly computed RMC value (should be saved separately)
        '''
        # first time computing RMC, set it to the max (=1) and store the loss
        if log_name not in self.losses:
            self.losses[log_name] = deque([], maxlen=self.rmc_lookback)
            rmc = 1
        
        # as long as len(losses) < rmc_lookback, RMC = 1
        elif len(self.losses[log_name]) < self.rmc_lookback:
            rmc = 1
        
        # this is the actual computation of RMC
        else:
            old_loss = self.losses[log_name][0]
            rmc = (old_loss - loss) / old_loss
            rmc = max(0, rmc)

        self.losses[log_name].append(loss)
        
        return rmc
    

    @abstractmethod
    def execute(self, models: list[torch.nn.Module], tasks: list[Task]) -> None:
        '''
        Abstract method to execute the training algorithm.
        This method requires a list of tasks and the models to execute them. This behavior may be
        changed in the future (e.g., using head indices to specify where to route tasks within a 
        model, or attaching heads to the tasks and sequentially passing through the backbone and 
        the head).

        Args:
            models (list[torch.nn.Module]): list of models to execute tasks with.
            tasks (list[Task]): list of tasks to execute.

        Raises:
            NotImplementedError: must create a subclass of this and implement the method.
        '''
        raise NotImplementedError()



class SequentialTraining(TrainingAlgorithm):    
    def __init__(self,
                epochs: int,
                optimizer_fn: Callable[..., torch.optim.Optimizer],
                optimizer_params: dict[str, any],
                **kwargs) -> None:
        '''
        Train a model one task at a time.
        This object will train a model for `epochs` epochs on each one of the given tasks, one task
        at a time, in the same order they are given. Note that accuracy on earlier tasks may degrade
        as new tasks are used. 

        Args:
            epochs (int): number of epochs to train for each task
        '''
        super(SequentialTraining, self).__init__(epochs, optimizer_fn, optimizer_params, **kwargs)
    

    def execute(self, backbone: torch.nn.Module, tasks: list[Task], epochs_passed: int) -> None:
        '''
        Train the backbone and all the heads in all the tasks.

        Args:
            backbone (torch.nn.Module): backbone model to train that extracts features.
            tasks (list[Task]): list of tasks with datasets and specialized heads to train.
            freeze_backbone (bool): whether to train
        '''
        if not self.optimizer_is_configured:
            self.configure_optimizer(backbone)
        
        # freeze the backbone if specified
        for param in backbone.parameters():
            param.requires_grad = not self.freeze_backbone

        # train one task at a time for given epochs
        for task in tasks:
            for epoch in trange(epochs_passed, self.epochs + epochs_passed, desc = task.name):
                backbone.train()
                is_last_batch = False
                losses = []

                # full pass over train dataset
                while not is_last_batch:
                    # foward + backward pass
                    self.optimizer.zero_grad()
                    is_last_batch, loss = task.train_step(backbone)

                    # backbone weights update
                    losses.append(loss)
                    self.optimizer.step()
                
                # optimizer update
                self.scheduler.step(sum(losses) / len(losses))

                # log / eval
                self.full_task_eval(backbone, task)
                self.indipendent_capacity_evaluation(backbone)
                self.data_dependent_capacity_evaluation(backbone, [t.testloader for t in tasks])
                self.algo_dependent_capacity_evaluation()
                self.logger.increase_step()

        return self.epochs


class AlternatingTraining(TrainingAlgorithm):
    def __init__(self,
                epochs: int,
                optimizer_fn: Callable[..., torch.optim.Optimizer],
                optimizer_params: dict[str, any],
                **kwargs) -> None:
        '''
        Train a model on multiple tasks, one dataset pass at a time.
        This object will train a model for `epochs` epochs, where each epoch trains a branch on one
        given task's entire dataset. Therefore, given 3 tasks, to make sure each task' dataset is
        passed 10 times, one should specify 30 epochs.

        Args:
            epochs (int): number of epochs to train for all task
            freeze_backbone (bool): whether to train the heads only (True) or also the backbone (False).
        '''
        super(AlternatingTraining, self).__init__(epochs, optimizer_fn, optimizer_params, **kwargs)
    
    def execute(self, backbone: torch.nn.Module, tasks: list[Task], epochs_passed: int) -> int:
        '''
        Train the backbone and all the heads in all the tasks.

        Args:
            backbone (torch.nn.Module): backbone model to train that extracts features.
            tasks (list[Task]): list of tasks with datasets and specialized heads to train.
        '''
        if not self.optimizer_is_configured:
            self.configure_optimizer(backbone)
        
        # freeze the backbone if specified
        for param in backbone.parameters():
            param.requires_grad = not self.freeze_backbone

        for epoch in trange(epochs_passed, self.epochs + epochs_passed, desc = 'Epochs', leave=False):
            for task in tasks:
                backbone.train()
                is_last_batch = False
                losses = []
                pbar = tqdm(total=len(task.trainloader), desc=task.name+' train')
                
                # full pass over train dataset
                while not is_last_batch:
                    # foward + backward pass
                    self.optimizer.zero_grad()
                    is_last_batch, loss = task.train_step(backbone)

                    # backbone weights update
                    losses.append(loss)
                    self.optimizer.step()
                    pbar.update(1)
                
                # optimizer update
                self.scheduler.step(sum(losses) / len(losses))
                pbar.close()
                
                self.full_task_eval(backbone, task)
                self.indipendent_capacity_evaluation(backbone)
                self.data_dependent_capacity_evaluation(backbone, [t.testloader for t in tasks])
                self.algo_dependent_capacity_evaluation()
            self.logger.increase_step()

        return self.epochs
