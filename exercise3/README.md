# Exercise 3 – Regularization & Recurrent Neural Networks

This exercise rounds out the NumPy framework with regularization techniques, batch
normalization, and recurrent layers. It also trains a LeNet-style CNN on MNIST using
the framework built up over the previous exercises.

## New components (`src_to_implement/`)

### Regularization & normalization (`Layers/`, `Optimization/`)
- **`Dropout.py`** – Inverted dropout layer (active in training, identity at test time).
- **`BatchNormalization.py`** – Batch normalization for both vector and image inputs,
  with running mean/variance estimates used during testing.
- **`Constraints.py`** (in `Optimization/`) – `L1_Regularizer` and `L2_Regularizer`
  weight penalties, integrated with the optimizers via a regularizer hook.

### Recurrent & extra activation layers (`Layers/`)
- **`RNN.py`** – Elman recurrent layer with hidden-state memorization
  (`memorize` flag), full backpropagation-through-time, and its own optimizer/weights.
- **`Sigmoid.py`** and **`TanH.py`** – activation functions used by the recurrent layer.

## Reused components

- **Layers**: `Base`, `FullyConnected`, `Conv`, `Pooling`, `Flatten`, `ReLU`,
  `SoftMax`, `Initializers`, `Helpers`.
- **Optimization**: `Optimizers` (now with a shared `Optimizer` base plus `Sgd`,
  `SgdWithMomentum`, `Adam`) and `Loss` (`CrossEntropyLoss`).
- **`NeuralNetwork.py`**: training/testing orchestration, now regularization-aware.

## Training & data

- **`TrainLeNet.py`** – Trains a LeNet architecture on the MNIST dataset.
- **`Data/`** – MNIST IDX archives (`train-*` and `t10k-*` images/labels).
- **`Models/`** – Saved/serialized models.

## Tests & results

`NeuralNetworkTests.py` covers the new layers and regularizers. Training results are
recorded in `log.txt`, comparing configurations on the UCI digits dataset, e.g.:

- Batch norm: up to ~98.7%
- Batch norm + L2: ~97–98%
- Adam: ~97%
- Dropout: ~96–97.5%
- L1 / L2 regularizers: ~95.5–97.7%

with ~94–98% accuracy on the Iris dataset across the various settings.

## Running

```bash
cd src_to_implement
python NeuralNetworkTests.py        # run the unit tests
python TrainLeNet.py                # train LeNet on MNIST
```

Dependencies are listed in `requirements.txt`.
