# TensorFlow Number Recognizer

A project for recognizing handwritten digits using a neural network built with TensorFlow and trained on the MNIST dataset. The project includes code for both training a model and using it to predict digits from new images.

---

## Features

- **Model Training:** Trains a neural network on the MNIST dataset (28x28 grayscale images of handwritten digits).
- **Model Architecture:** Simple sequential model with Dense and Dropout layers, using ReLU activation and Adam optimizer.
- **Visualization:** Plots training and validation loss and accuracy.
- **Prediction:** Loads the trained model and predicts digits from new images.
- **Preprocessing:** Includes preprocessing steps to resize and grayscale images for input.
- **TensorBoard Support:** Integrates TensorBoard for advanced training visualization.

---

## Getting Started

### Prerequisites

- Python 3.x
- TensorFlow
- TensorFlow Datasets
- NumPy
- Matplotlib
- Pillow (PIL)

You can install the required packages using:

```bash
pip install tensorflow tensorflow-datasets numpy matplotlib pillow
```

---

### Training the Model

To train the digit recognition model, run:

```bash
python modello_cifre.py
```

This script will:
- Download the MNIST dataset.
- Normalize and batch the data.
- Train the model for 40 epochs.
- Save the trained model to `./classificatore_cifre/mio_modello.keras`.
- Plot training/validation loss and accuracy.

---

### Using the Model for Prediction

To use the trained model to predict digits on new images, run:

```bash
python usa_modello_cifre.py
```

The script:
- Loads the saved model.
- Preprocesses a set of images (by default, it looks for images in `classificatore_cifre/test/mio_numero_{i}.jpg`).
- Predicts the digit class and prints the result with confidence.

**Note:** Images must be 28x28 pixels or will be resized automatically. Images should be formatted similarly to MNIST (white digit on black background is preferable).

---

## File Structure

- `modello_cifre.py`: Script to train and evaluate the neural network on MNIST.
- `usa_modello_cifre.py`: Script to load the trained model and predict digits from new images.
- `classificatore_cifre/`: Directory where the trained model and test images are stored.

---

## Example

### Training Output

- Loss and accuracy plots will be displayed after training.
- Model will be saved to `classificatore_cifre/mio_modello.keras`.

### Prediction Output

Sample output:
```
Classe predetta: 7 con confidenza: 0.98%
Classe predetta: 3 con confidenza: 0.95%
...
```

---

## Customization

- To use your own images, place them in the `classificatore_cifre/test/` folder and update the script if you use different filenames or paths.
- You can modify the model architecture in `modello_cifre.py` to experiment with different neural network designs.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

- [TensorFlow](https://www.tensorflow.org/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [TensorFlow Datasets](https://www.tensorflow.org/datasets)
