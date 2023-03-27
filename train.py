import argparse
from util.model import Model

parser = argparse.ArgumentParser()

parser.add_argument('--input_shape_structured', type=str, default='3', help='Input shape for structured data (comma-separated).')
parser.add_argument('--input_shape_unstructured', type=str, default='None,512,512,1', help='Input shape for unstructured data (comma-separated; D, H, W, C).')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training.')
parser.add_argument('--validation_split', type=float, default=0.2, help='Fraction of the training data to be used as validation data.')
parser.add_argument('--save_weights', action='store_true', help='Whether to save the model weights after training.')
parser.add_argument('--save_weights_path', type=str, default='./model_weights.h5', help='Path for saving model weights.')
parser.add_argument('--load_weights', action='store_true', help='Whether to load model weights before training.')
parser.add_argument('--load_weights_path', type=str, default='./model_weights.h5', help='Path for loading model weights.')
parser.add_argument('--dataset_path', type=str, default='./dataset.npz', help='Path for the dataset.')

args = parser.parse_args()

if __name__ == '__main__':

    input_shape_structured = tuple(map(int, args.input_shape_structured.split(',')))
    input_shape_unstructured = tuple(map(lambda x: None if x == 'None' else int(x), args.input_shape_unstructured.split(',')))

    model = Model(input_shape_structured, input_shape_unstructured)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.plot()

    # Load the dataset from the specified path
    x_train, y_train, x_test, y_test = load_dataset(args.dataset_path)

    if args.load_weights:
        model.load_weights(args.load_weights_path)

    history = model.train(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs, validation_split=args.validation_split)

    loss, accuracy = model.test(x_test, y_test)
    print(f"Test loss: {loss}, Test accuracy: {accuracy}")

    if args.save_weights:
        model.save_weights(args.save_weights_path)
