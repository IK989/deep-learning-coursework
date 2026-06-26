# Exercise 1 – Feed-Forward Neural Network (Basic Framework)

This exercise builds a small, modular deep-learning framework from scratch in NumPy.
It introduces the layer/optimizer abstractions that all later exercises extend, and
uses them to train a fully connected classifier with backpropagation.

See `Description.pdf` and `1_BasicFramework.pdf` for the full task description.

## Architecture

`NeuralNetwork.py` ties everything together: it holds a list of layers, a loss layer,
an optimizer, and a weight initializer. It exposes `forward()`, `backward()`,
`train(iterations)`, and `test(input_tensor)`, wiring the data layer, trainable
layers, and loss into a full training loop.

## Components (`src_to_implement/`)

### Layers (`Layers/`)
- **`Base.py`** – `BaseLayer`, the common base class (e.g. the `trainable` flag).
- **`FullyConnected.py`** – Affine layer with `forward`/`backward`, weight gradients,
  and an optimizer hook.
- **`ReLU.py`** – Rectified Linear Unit activation.
- **`SoftMax.py`** – SoftMax activation for producing class probabilities.
- **`Initializers.py`** – Weight initialization schemes: `Constant`, `UniformRandom`,
  `Xavier`, and `He`.
- **`Helpers.py`** – Utilities and datasets used by the tests.

### Optimization (`Optimization/`)
- **`Optimizers.py`** – `Sgd`, `SgdWithMomentum`, and `Adam`.
- **`Loss.py`** – `CrossEntropyLoss`.

## Running

```bash
cd src_to_implement
python NeuralNetworkTests.py        # run the provided unit tests
```

Dependencies are listed in `requirements.txt` (NumPy 1.26.4, scikit-learn, scipy,
matplotlib, scikit-image, etc.).
