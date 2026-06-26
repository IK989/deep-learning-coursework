# Exercise 0 – Python & NumPy Fundamentals

This exercise builds the groundwork for the rest of the course: generating data with
NumPy and writing a reusable image data generator. The emphasis is on vectorized
NumPy operations (no explicit Python loops over pixels) and clean, testable classes.

## Contents

### `pattern.py` – Synthetic pattern generation
Three classes, each exposing a `draw()` method (returns a copy of the pattern) and a
`show()` method (displays it with matplotlib):

- **`Checker(resolution, tile_size)`** – Builds a black/white checkerboard by tiling a
  `[[0, 1], [1, 0]]` base block with `np.tile` and `np.repeat`. The top-left tile is
  black (0). Requires `resolution` to be divisible by `2 * tile_size`.
- **`Circle(resolution, radius, position)`** – Draws a binary circle using a
  `np.meshgrid` coordinate grid and the Euclidean distance formula
  (`distance <= radius`). Returns a boolean mask.
- **`Spectrum(resolution)`** – Creates an RGB color spectrum where the red channel
  increases left-to-right, the green channel increases top-to-bottom, and the blue
  channel is the inverse of red.

### `generator.py` – `ImageGenerator`
A batch data generator for the CIFAR-style `.npy` image dataset that supports:

- **Batching** via `next()`, returning `(images, labels)` and wrapping around at the
  end of an epoch.
- **Resizing** images to a target `image_size` using `skimage.transform.resize`.
- **Shuffling** the dataset (re-shuffled at the start of each new epoch).
- **Augmentation** (`_augment`): random horizontal mirroring and random 90°/180°/270°
  rotations.
- **Epoch tracking** via `current_epoch()` and label-to-name lookup via
  `class_name(label)` using the 10-class CIFAR `class_dict`.
- **`show()`** to plot a full batch with class-name titles.

### Supporting files
- `main.py` – Driver script that exercises the pattern and generator classes.
- `NumpyTests.py` – Unit tests for the patterns and the image generator.
- `data/` – Image data and labels (JSON) used by the generator.
- `reference_arrays/` – Reference outputs used by the tests.

## Running

```bash
python NumpyTests.py        # run the unit tests
python main.py              # generate and display the patterns / batches
```

Key dependencies: `numpy`, `matplotlib`, `scikit-image`.
