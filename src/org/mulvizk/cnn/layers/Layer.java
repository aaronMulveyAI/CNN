package org.mulvizk.cnn.layers;

import java.util.ArrayList;
import java.util.List;

public abstract class Layer {

    protected Layer nextLayer;
    protected Layer previousLayer;

    public abstract double[] getOutput(List<double[][]> input);
    public abstract double[] getOutput(double[] input);

    public abstract void backPropagation(List<double[][]> diffLoss_diffOut);
    public abstract void backPropagation(double[] diffLoss_diffOut);

    public abstract int getOutputLength();
    public abstract int getOutputRows();
    public abstract int getOutputCols();
    public abstract int getOutputElements();

    public double[] matrixToVector(List<double[][]> input){

        int lenght = input.size();
        int rows = input.get(0).length;
        int cols = input.get(0)[0].length;
        double [] vector = new double[lenght * rows * cols];
        int count = 0;

        for (int i = 0; i < lenght; i++) {
            for (int j = 0; j < rows; j++) {
                for (int k = 0; k < cols; k++) {
                    vector[count] = input.get(i)[j][k];
                    count++;
                }
            }
        }
        return vector;
    }

    public List<double[][]> vectorToMatrix(double[] input, int lenght, int rows, int cols){

        List<double[][]> output = new ArrayList<>();
        int count = 0;

        for (int k = 0; k < lenght; k++){

            double [][] matrix = new double[rows][cols];

            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    matrix[i][j] = input[count];
                    count++;
                }
            }
            output.add(matrix);
        }
        return output;
    }


    public void setNextLayer(Layer layer) {
        nextLayer = layer;
    }

    public void setPreviousLayer(Layer layer) {
        previousLayer = layer;
    }
}
