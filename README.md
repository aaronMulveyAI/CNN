# Java Convolutional Neural Network (CNN) Library

This library provides a Java-based implementation of a Convolutional Neural Network (CNN) for image classification tasks. It includes modular components for building, training, and testing neural networks using customizable layers and configurations.

## Features
- **Data Processing**: Tools for reading and writing image data in CSV format.
- **Convolutional Layers**: Perform convolution operations with customizable filters and strides.
- **Max Pooling Layers**: Downsample data to reduce spatial dimensions.
- **Fully Connected Layers**: Traditional dense layers with backpropagation.
- **Neural Network Building**: Flexible architecture design with the `NetworkBuilder` class.
- **Training and Testing**: Train networks with datasets and evaluate performance.

## Installation
1. Clone the repository or download the source files.
2. Include the source files in your Java project.
3. Ensure you have the necessary Java development environment (JDK 8+).

## Usage

### Example: Training and Testing a CNN
The following example demonstrates how to load data, build a CNN, and train/test it:

```java
import org.mulvizk.cnn.Main;

public class Example {
    public static void main(String[] args) {
        Main.main(args);
    }
}
```

This runs the `Main` class, which:
1. Loads training and testing datasets from CSV files.
2. Builds a CNN with a convolutional layer, a max pooling layer, and a fully connected layer.
3. Trains the CNN and evaluates its performance on test data.

### Dataset Format
Datasets must be in CSV format, where each row represents an image. The first column is the label, and the remaining columns represent pixel values.

### Building a Network
Use the `NetworkBuilder` class to design a network:
```java
NetworkBuilder builder = new NetworkBuilder(28, 28, 256 * 10);
builder.addConvolutionLayer(8, 5, 1, 0.1, 123);
builder.addMaxPoolLayer(3, 2);
builder.addFullyConnectedLayer(10, 0.9, 123);
NeuralNetwork nn = builder.build();
```

### Training the Network
Train the network using:
```java
nn.train(dataTrain);
```

### Testing the Network
Evaluate performance with:
```java
float accuracy = nn.test(dataTest);
System.out.println("Test Accuracy: " + (accuracy * 100) + "%");
```

## API Overview

### Data Handling
- **`Reader`**: Reads datasets from CSV files into `Image` objects.
- **`Writter`**: Exports data (e.g., weights) to CSV files.

### Layers
- **`ConvolutionLayer`**: Implements convolutional operations with filters.
- **`MaxPoolLayer`**: Performs max pooling.
- **`FullyConnectedLayer`**: Implements fully connected operations with ReLU activation.

### Neural Network
- **`NetworkBuilder`**: Simplifies network architecture design.
- **`NeuralNetwork`**: Manages training, testing, and inference.

### Utility Classes
- **`MatrixUtils`**: Provides basic matrix operations.
- **`Image`**: Represents an image with pixel data and label.

## Contributing
Contributions are welcome! Please submit a pull request with a clear description of changes and improvements.

