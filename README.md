# Connected Hidden Neurons (CHNNet): An Artificial Neural Network for Rapid Convergence


This `README.md` file contains the instructions for implementing the experiments accompanying the paper.


## Dataset
The datasets are sourced from [Penn Machine Learning Benchmarks (PMLB)](https://epistasislab.github.io/pmlb/). A guide to retrieve the datasets is given in [PMLB python documentation](https://epistasislab.github.io/pmlb/using-python.html)
### Regression
- **Fried:** This dataset is automatically imported from PMLB while executing the code.
- **BNG PWLinear:** This dataset is automatically imported from PMLB while executing the code.
- **Breast Tumor:** This dataset is automatically imported from PMLB while executing the code.
### Binary classification
- **Adult:** This dataset is automatically imported from PMLB while executing the code.
- **Magic:** This dataset is automatically imported from PMLB while executing the code.

### Multiclass classification

- **Connect:** This dataset is automatically imported from PMLB while executing the code.
- **Fars:** This dataset is automatically imported from PMLB while executing the code.
- **Letter:** This dataset is automatically imported from PMLB while executing the code.
- **Mnist:** This dataset is automatically imported from PMLB Datasets while executing the code.
- **Sleep:** This dataset is automatically imported from PMLB Datasets while executing the code.


## Environment Setup
All the experiments accompanying the paper have been conducted using TensorFlow 2.10. A detailed guideline for installing TensorFlow with pip can be found on their official [website](https://www.tensorflow.org/install/pip).


## Requirements

The experiments are carried out in a Python 3.8 environment. The following additional packages are required to run the tests:
- tensorflow (version 2.15.0)
- keras (version 2.15.0)
- matplotlib (version 3.5.1)
- pandas (version 2.1.4)
- scikit-learn (version 1.3.2)
- pmlb (version 1.0.1)


The dependencies can be installed manually or by using the following command:
```
pip install -r ./requirements.txt
```
It is recommended to use a virtual environment for installing all the modules.

### Potential Errors
While using tensorflow-dataset, you can encounter the following error:
```
importError: cannot import 'builder' from google.protobuf.internal
```
To fix this error, you can install protobuf version 3.20 using the following command:
```
pip install protobuf==3.20
```

## Code Explanation

### CHN Layer


The CHN layer is coded in the `CHNLayer.py` file using `Layer` superclass of Keras. The variables named `kernel_Input_Units` and `kernel_Hidden_Units` in `build` function represent the two sets of weights mentioned in the paper.


The `call` method defines the forward pass of the layer and can handle different types of inputs. The backpropagation of the layer is handled by tensorflow itself. TensorFlow's automatic differentiation mechanism has been used to calculate the gradients during backpropagation.


### Test Files

The py files named {dataset_name}.py holds the codes for the tests on that respective dataset. By executing the codes in these files, the test results for the respective dataset can be generated.

### Parameters

The following adjustable parameters allow for customization of the model's architecture, training duration, and optimization strategy based on the specific task and dataset:


1. `num_seeds`: represents the total number of seed for each architecture
2. `arch`: represents the total number of architecture
3. `epochs`: represents the number of epochs for training.
4. `batchSize`: represents the size of each batch for training.
5. `layers`: represents the total number of layers
6. `FNN_hn`: represents the number of hidden neurons in the n<sup>th</sup> hidden layer of the Dense layer.
7. `CHN_hn`: represents the number of hidden neurons in the n<sup>th</sup> hidden layer of the CHNLayer.
8. `isEqual`: `True` represents equal number of parameter
9. `init`: represent the architecture number range in the graphs
10. `loss`: determines the objective function used to measure the model's performance and guide its learning during training.
11. `optimizer`: determines the algorithm used to optimize the neural network model during training.



### Results

- When the training is complete, the model summary and statistical test results are displayed on the terminal.
- The loss graphs for each seed are displayed in separate windows afterward.