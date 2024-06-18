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
import os
import torch
from configparser import ConfigParser

# inter-class dependencies only for type annotations
from emtl.algorithms import TrainingAlgorithm
from emtl.tasks import Task


class Trainer:
    backbone: torch.nn.Module
    tasks: list[Task]
    use_mlflow: bool
    mlflow_tags: dict[str, any]
    mlflow_params: dict[str, any]
    run_id: str
    epochs_passed: int
    
    def __init__(self, 
                 backbone: torch.nn.Module, 
                 tasks: list[Task], 
                 device: str = None, 
                 use_mlflow: bool = None,
                 config: str = '',
                 **kwargs
        ) -> None:
        '''
        Create a coordinator object Trainer.
        Trainer objects are responsible for wrapping together tasks, models, and training algos. 
        The trainer is the one that connects the backbone model to the tasks heads (saving those
        models), and that launches the training of the algorithm, passing in tasks and models.

        Currently a todo feature, the config file allows one to specify parameters for the trainer
        other than through parameters (e.g., tags for the MLFlow run).

        Args:
            backbone (torch.nn.Module): instantiated backbone model to train
            tasks (list[Task]): list of tasks with datasets and specialized heads to train
            algorithm (TrainingAlgorithm): implementation of a training algorithm to consume tasks
            device (str, optional): cpu or cuda:id device to push all classes to. Defaults to 'cpu'.
            use_mlflow (bool, optional): set True to use MLFlow (must be installed). Defaults to False.
            config (str, optional): path to a configuration file to replace params. Defaults to ''.
        '''
        self.tasks = tasks
        self.epochs_passed = 0
        self.run_id = None
        
        # assign instance variables from configuration or parameters (parser empty by default if
        # filepath is not specified, so no errors are thrown, and defaults are returned)
        this_path = os.path.dirname(__file__)
        default_config_path = os.path.join(this_path, os.pardir, 'configs', 'trainer', 'default.ini')

        parser = ConfigParser()
        parser.read(default_config_path)    # default values
        parser.read(config)                 # values from given config file

        # update parser with supplied arguments, if any
        if use_mlflow is not None:      parser.set('mlflow', 'enabled', str(use_mlflow))
        if device:                      parser.set('pytorch', 'device', device)

        self.use_mlflow = parser.getboolean('mlflow', 'enabled')
        device = parser.get('pytorch', 'device')

        # save models
        self.backbone = backbone.to(device)

        # connect to MLFlow if it was specified
        if self.use_mlflow:
            self.setup_mlflow(parser, **kwargs)
    

    def setup_mlflow(self, 
                     parser: ConfigParser,
                     mlflow_database: str = None,
                     mlflow_experiment: str = None,
                     mlflow_tags: dict[str, any] = None, 
                     mlflow_params: dict[str, any] = None
        ) -> None:
        '''
        Setup and connect to an MLFlow instance.
        This method will load an MLFlow database for experiment tracking (or create a new one if the 
        file specified does not exist), and load/create the specified experiment. One can specify
        tags and parameters to attach to the experiment.

        Note: all configurations of this method can be set via programmed parameters or through a
        configuration file.

        Args:
            parser (ConfigParser): loaded parser with default and optional configuration.
            mlflow_database (str, optional): path to the MLFlow DB file. Defaults to 'sqlite:///mlflow.db'.
            mlflow_experiment (str, optional): Name of the experiment. Defaults to 'Default'.
            mlflow_tags (dict[str, any], optional): dictionary of tags. Defaults to {}.
            mlflow_params (dict[str, any], optional): dictionary of parameters. Defaults to {}.
        '''
        import mlflow, ast

        # first update parser with any passed arguments
        if mlflow_database:     parser.set('mlflow', 'database', mlflow_database)
        if mlflow_experiment:   parser.set('mlflow', 'experiment', mlflow_experiment)

        # then, set/use the values
        mlflow.set_tracking_uri(parser.get('mlflow', 'database'))
        mlflow.set_experiment(parser.get('mlflow', 'experiment', fallback='Default'))
        self.mlflow_tags = mlflow_tags if mlflow_tags else \
            ast.literal_eval(parser.get('mlflow', 'tags', fallback='{}'))
        self.mlflow_params = mlflow_params if mlflow_params else \
            ast.literal_eval(parser.get('mlflow', 'params', fallback='{}'))

    def assert_models_and_tasks_are_valid(self) -> None:
        '''
        Sanity check to make sure data from the tasks' loader can go through the backbone & heads.
        More rigorous testing would require passing in the input and output shapes at execution time,
        which is not user-friendly.
        '''
        # to test that models and tasks work as expected, for a sanity check, let's make one fw pass
        with torch.no_grad():
            for task in self.tasks:
                # train set
                (inputs, _), _ = next(task.train_producer())
                features = self.backbone(inputs)
                task.head(features)
                
                # test set
                (inputs, _), _ = next(task.test_producer())
                features = self.backbone(inputs)
                task.head(features)
    
    def launch(self, algorithm: TrainingAlgorithm, run_id: str = None, reset_optimizers: bool = True) -> str:
        '''
        Launch the training algorithm, optionally in a new MLFLow session (whose ID is returned).

        Args:
            run_id (str, optional): can reuse an existing MLFlow run. Defaults to None.

        Returns:
            str: the run ID of this MLFlow experiment (if any).
        '''
        if reset_optimizers:
            algorithm.reset_optimizers(self.backbone)
            
            for task in self.tasks:
                task.reset_optimizers()

        # MLFlow is disabled
        if not self.use_mlflow:
            epochs = algorithm.execute(self.backbone, self.tasks, self.epochs_passed)
            self.epochs_passed += epochs
            return

        # MLFLow is enabled, and there may be problems, so we need to close the run anyways
        try:
            import mlflow

            # use the specified run_id if supplied, otherwise resume the previous run, if it exists
            run_id = run_id if run_id else self.run_id

            with mlflow.start_run(run_id=run_id, tags=self.mlflow_tags) as run:
                mlflow.log_params(self.mlflow_params)
                epochs = algorithm.execute(self.backbone, self.tasks, self.epochs_passed)
                self.epochs_passed += epochs
                self.run_id = run.info.run_id
                return run.info.run_id

        except Exception:
            mlflow.end_run()
            raise

