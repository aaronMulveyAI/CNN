package org.mulvizk.cnn.data;

import java.util.List;

public class MainReader {

    public static final String PATH = "/Users/aaronmulvey/Documents/Proyects/Java/Neural Networks/CNN/src/org/mulvizk/data/";
    public static final String TEST_FILE ="mnist_test.csv";
    public static final String TRAIN_FILE ="mnist_train.csv";

    public static void main(String[] args) {

        List<Image> dataTrain = new Reader().readCSV(PATH + TRAIN_FILE);
        List<Image> dataTest = new Reader().readCSV(PATH + TRAIN_FILE);
        dataTrain.forEach(System.out :: println);
        dataTest.forEach(System.out :: println);

        double[][] data = dataTest.get(0).getData();

        for (int i = 0; i < data.length; i++) {
            System.out.println();
            for (int j = 0; j < data[0].length; j++) {
                System.out.print(data[i][j]);
            }
        }
    }
}
