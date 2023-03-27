import argparse
from util.model import Model

parser = argparse.ArgumentParser()

parser.add_argument('--input_shape_structured', type=str, default='3', help='Input shape for structured data (comma-separated).')
parser.add_argument('--input_shape_unstructured', type=str, default='None,512,512,1', help='Input shape for unstructured data (comma-separated; D, H, W, C).')
parser.add_argument('--weights_path', type=str, default='./model_weights.h5', help='Path for loading the trained weights.')
parser.add_argument('--test_dataset_path', type=str, default='./test_dataset.npz', help='Path for loading the test dataset.')

args = parser.parse_args()

if __name__ == '__main__':
    input_shape_structured = tuple(map(int, args.input_shape_structured.split(',')))
    input_shape_unstructured = tuple(map(lambda x: None if x == 'None' else int(x), args.input_shape_unstructured.split(',')))

    model = Model(input_shape_structured, input_shape_unstructured)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Load the trained weights from the specified path
    model.load_weights(args.weights_path)

    # Load the test dataset from the specified path
    x_test, y_test = load_test_dataset(args.test_dataset_path)

    loss, accuracy = model.test(x_test, y_test)
    print(f"Test loss: {loss}, Test accuracy: {accuracy}")
