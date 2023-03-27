# Predicting the severity of obstructive sleep apnea based on paranasal sinus computed tomography scan using multimodal deep learning models

Obstructive Sleep Apnea (OSA) is a common sleep disorder that can lead to serious health problems if left untreated. Paranasal sinus computed tomography (CT) scans have shown promise in predicting OSA severity, but manual interpretation is time-consuming and error-prone. 

This project implements multimodal deep learning models to predict the severity of OSA based on paranasal sinus CT scans. The models take both structured and unstructured data as inputs, and are trained and tested using the provided dataset.

## Requirements

| Package                 | Version       |
|------------------------|---------------|
| Python                 | \>= 3.6 (3.10 recommended)        |
| TensorFlow             | \>= 2.8.0 (2.9.0 recommended)         |
| OpenCV                 | Lastest       |
| tqdm                   | Lastest       |
| Pandas                 | Lastest       |
| PyDICOM                | Lastest       |

To install the required packages, run the following command:

```
pip install -r requirements.txt
```


## Usage

### Training

To train the models, run the following command:


```
python train.py --input_shape_structured 3 --input_shape_unstructured None,512,512,1 --batch_size 32 --epochs 10 --validation_split 0.2 --save_weights --save_weights_path ./model_weights.h5 --load_weights --load_weights_path ./model_weights.h5 --dataset_path ./dataset.npz
```


The arguments are:

| Argument                  | Description                                                                                   |
|---------------------------|-----------------------------------------------------------------------------------------------|
| `input_shape_structured`   | Input shape for structured data (comma-separated).                                           |
| `input_shape_unstructured` | Input shape for unstructured data (comma-separated; D, H, W, C).                             |
| `batch_size`               | Batch size for training.                                                                      |
| `epochs`                   | Number of epochs for training.                                                                |
| `validation_split`         | Fraction of the training data to be used as validation data.                                  |
| `save_weights`             | Whether to save the model weights after training.                                             |
| `save_weights_path`        | Path for saving model weights.                                                                |
| `load_weights`             | Whether to load model weights before training.                                                |
| `load_weights_path`        | Path for loading model weights.                                                               |
| `dataset_path`             | Path for the dataset.                                                                         |

### Testing

To test the models, run the following command:


```
python test.py --input_shape_structured 3 --input_shape_unstructured None,512,512,1 --weights_path ./model_weights.h5 --test_dataset_path ./test_dataset.npz
```


The arguments are:

| Argument                  | Description                                                                                   |
|---------------------------|-----------------------------------------------------------------------------------------------|
| `input_shape_structured`   | Input shape for structured data (comma-separated).                                           |
| `input_shape_unstructured` | Input shape for unstructured data (comma-separated; D, H, W, C).                             |
| `weights_path`             | Path for loading the trained weights.                                                         |
| `test_dataset_path`        | Path for loading the test dataset.                                                            |

## Data Availability

The data that support the ﬁndings of this study are available from the corresponding author, Jin Youp Kim (kjyoup0622@gmail.com), upon reasonable request.
