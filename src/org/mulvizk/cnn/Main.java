package org.mulvizk.cnn;

import org.mulvizk.cnn.data.Image;
import org.mulvizk.cnn.data.Reader;
import org.mulvizk.cnn.data.Writter;
import org.mulvizk.cnn.layers.FullyConnectedLayer;
import org.mulvizk.cnn.network.NetworkBuilder;
import org.mulvizk.cnn.network.NeuralNetwork;

import java.util.List;
import static java.util.Collections.shuffle;

public class Main {

    public static final String PATH = "/Users/aaronmulvey/Documents/Proyects/Java/Neural Networks/CNN/src/org/mulvizk/data/";
    public static final String TEST_FILE ="mnist_test.csv";
    public static final String TRAIN_FILE ="mnist_train.csv";

    public static void main(String[] args) {

        long SEED = 123;

        System.out.println("Starting loading data...");
        List<Image> dataTrain = new Reader().readCSV(PATH + TRAIN_FILE);
        List<Image> dataTest = new Reader().readCSV(PATH + TEST_FILE);

        System.out.println("Training data size = " + dataTrain.size());
        System.out.println("Testing data size = " + dataTest.size());


        NetworkBuilder builder = new NetworkBuilder(28, 28, 256 * 10);
        builder.addConvolutionLayer(8, 5, 1, 0.1, SEED);
        builder.addMaxPoolLayer(3, 2);
        builder.addFullyConnectedLayer(10, 0.9, SEED);


        NeuralNetwork nn = builder.build();

        float rate = nn.test(dataTest);

        System.out.println("Pre training rate: " + rate * 100 + "%");

        int epochs = 0;

        for (int i = 0; i < epochs; i++){
            shuffle(dataTrain);
            nn.train(dataTrain);
            rate = nn.test(dataTest);
            System.out.println("Success rate after epoch " + (i + 1) + " " + rate * 100 + "%");

        }

        FullyConnectedLayer a = (FullyConnectedLayer) nn.getLayers().get(2);
        Writter.writeCSV(a.getWeights(), "a.csv");
    }

}
