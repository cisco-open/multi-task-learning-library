<!-- # Easy Multi-Task Learning -->
<p align="center" width="100%">
    <img width="150px" src="https://i.postimg.cc/HWzrbmZX/Screenshot-20230301-054816.png">
</p>
<h1 style="text-align: center;">Easy Multi-Task Learning Library</h1>
<p style="text-align: center;">https://wwwin-github.cisco.com/pages/chflemin/easy-mulit-tast-learning-library/</p>

Easy Multi-Task Learning Library (EMTL) is a framework to simplify the prototyping and training processes for multi-task computer vision ML models. EMTL provides a set of interfaces and tools to modularize CV tasks and datasets into consumable objects and standardize models. These modularized objects are then assembled and given to a parameterized trainer, which will produce a complete trained model.

## Setup
The minimal installation requirements for this library are `PyTorch` and `tqdm`. Additionally, one
may install MLFlow to track experiments, which is fully integrated into this library.
```bash
# PyTorch CPU
pip3 install torch tqdm virtualenv

# PyTorch GPU
pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu117 tqdm virtualenv

# Optional MLFlow
pip3 install mlflow
```

### Run on a remote Jupyter Server
Here we provide a basic tutorial on how to spin up a Jupyter Server instance on a remote server to 
offload intensive computation when using the EMTL library. First of all, one needs a remote machine,
possibly equipped with GPUs or TPUs, where Anaconda is installed (installation instructions here
https://docs.anaconda.com/anaconda/install/linux/). 
Then, to connect it to a local Jupyter notebook in VS Code:
``` bash
# 1. SSH into the remote machine and forward port 8888 (for Jupyter)
ssh -L 8888:localhost:8888 user@address

# 2. Create & activate a new Anaconda virtual environment
virtualenv -p $(which python3) emtl_environment 
virtualenv activate emtl_environment
pip3 install torch tqdm mlflow jupyterlab

# 3. Spin up a Jupyter Server instance
jupyter server --no-browser --port=8888
```
4. Connect Jupyer Server to a local VS Code
    1. The last shell command will return *https* addresses with token. Copy one (the whole address).
    2. Open a notebook in local VS Code, and at the screen bottom select *Jupyter Server: local*.
    3. In the popup menu, select *Existing* and paste the copied url. Press *Enter* twice.
    4. You should now be connected to the remote server.
5. Switch to the correct kernel
    1. In the top-right part of the screen, where it says *kernel*, click it.
    2. In the popup menu, look for the option that mentions *server* or *remote*; click it.
    3. You should now be connected to the remote kernel. You can now run notebooks.

### Experiment Tracking with MLFlow
Here we show a basic use case of EMTL supported by MLFlow. We create an SQLite database to keep
track of experiments, and serve it through port 5000 of the remote host (that we forward locally).
Assuming one has already setup the `emtl_environment` as described above on the remote machine, do
the following:
```bash
# connect to remote server and forward ports (5000 for MLFLow, 8888 for Jupyter)
ssh -L 8888:localhost:8888 -L 5000:localhost:5000 user@address
conda activate emtl_environment

# spin up mlflow
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Then, connecting to http://localhost:5000/, you can find the UI of MLFlow with the experiments. To
connect to MLFLow from code and run a new experiment, do the following (in Python):
```python
import mlflow
mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow.set_experiment('EMTL Example')
```

To stop its execution, try:
```bash
fuser -k 5000/tcp
```
You can find more use cases in the examples and demos provided in this repository.


## Design
EMTL requires a Machine Learning Practitioner (MLP) to write their code in a *modular fashion*: models, datasets, learners, and algorithms are independent of one another, but can interact through the inferfaces and decorators provided by EMTL.

EMTL provides a useful pipeline to handle the creation and training of a multi-task model, with the following steps:
1. Define and validate the backbone model
2. For each taks to learn:
    2a. Create and validate the dataset
    2b. Define and validate the criterion
    2c. Define and validate the specialized head
    2d. Speicfy optional metadata
3. Choose a learning algorithm
    3a. EMTL will generate the learners
4. Launch the training
    4a. Log intermediate metrics (to console, file, or MLFlow DB)
5. Save the produced artifacts (trained backbone and heads)

EMTL validates models, datasets, and tasks through the *decorator* design pattern, ensuring that all modules are written according to its specification. Helpful insights are provided upon mismatch.

### Code Architecture
The project is structured so that we have an "umbrella" trainer class that requires a (backbone) model, a non-empty set of tasks, and a training algorithm. Each of these is one module (the tasks are a list of modules), and exists indipendently from the others (no class/function has imports from one to the other). We use a `config.ini` file to specify global configuration parameters (e.g., PyTorch's configurations, using MLFlow, etc.)

## Workflows
Here we list all pipelines to make and push changes, generate documentation, and run tests.

### Tests
Currently work in progress.
Tests are collected in the `tests` folder. To run tests, execute from the root of the project the
command:
```bash
python -m unittest -v
```
## Contributing & Future Directions

If you wish to contribute or suggest any additional funtionalities, please check out [Contributing Guidelines](/CONTRIBUTING.md)

## License

[Apache License 2.0](LICENSE).
