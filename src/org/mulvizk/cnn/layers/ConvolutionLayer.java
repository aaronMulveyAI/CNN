package org.mulvizk.cnn.layers;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import static org.mulvizk.cnn.data.MatrixUtils.*;

public class ConvolutionLayer extends Layer{

    private long SEED;
    private List<double[][]> filters;
    private int filterSize;
    private int stepSize;
    private int inLenght;
    private int inRows;
    private int inCols;
    private List<double[][]> lastInput;
    private double learningRate;

    public ConvolutionLayer(int filterSize, int n, double learningRate, int stepSize,
                            int inLength, int inRows, int inCols, long SEED) {
        this.filterSize = filterSize;
        this.stepSize = stepSize;
        this.learningRate = learningRate;
        this.inLenght = inLength;
        this.inRows = inRows;
        this.inCols = inCols;
        this.SEED = SEED;
        this.generateFilters(n);
    }

    public List<double[][]> convolutionalForwardPass(List<double[][]> input){

        lastInput = input;

        List<double[][]> output = new ArrayList<>();
        for (int i = 0; i < input.size(); i++) {

            for (double[][] filter: filters) {
                output.add(convolve(input.get(i), filter, stepSize));
            }
        }

        return output;
    }

    private double[][] convolve(double[][] input, double[][] filter, int stepSize) {

        int outRows = (input.length - filter.length) / stepSize + 1;
        int outCols = (input[0].length - filter[0].length) / stepSize + 1;

        int inRows = input.length;
        int inCols = input[0].length;

        int filterRows = filter.length;
        int filterCols = filter[0].length;

        double[][] output = new double[outRows][outCols];

        int outRow = 0;
        int outCol;

        for (int i = 0; i <= (inRows - filterRows); i += stepSize) {
            outCol = 0;

            for (int j = 0; j <= (inCols - filterCols); j += stepSize) {

                double sum = 0;

                for (int k = 0; k < filterRows; k++) {
                    for (int l = 0; l < filterCols; l++) {
                        int rowIndex = i + k;
                        int colIndex = j + l;

                        double value = filter[k][l] * input[rowIndex][colIndex];
                        sum += value;
                    }
                }

                output[outRow][outCol] = sum;
                outCol++;
            }
            outRow++;
        }

        return output;
    }

    private double[][] fullConvolution(double[][] input, double[][] filter) {

        int outRows = (input.length + filter.length) + 1;
        int outCols = (input[0].length + filter[0].length) + 1;

        int inRows = input.length;
        int inCols = input[0].length;

        int filterRows = filter.length;
        int filterCols = filter[0].length;

        double[][] output = new double[outRows][outCols];

        int outRow = 0;
        int outCol;

        for (int i = - filterRows + 1; i < inRows; i++) {
            outCol = 0;

            for (int j = - filterCols + 1; j < inCols; j++) {

                double sum = 0d;

                for (int k = 0; k < filterRows; k++) {
                    for (int l = 0; l < filterCols; l++) {
                        int rowIndex = i + k;
                        int colIndex = j + l;

                        if(rowIndex >= 0 && colIndex >= 0 && rowIndex < inRows && colIndex < inCols){
                            double value = filter[k][l] * input[rowIndex][colIndex];
                            sum += value;
                        }
                    }
                }

                output[outRow][outCol] = sum;
                outCol++;
            }
            outRow++;
        }

        return output;
    }


    public double[][] spaceArray(double[][] input){
        if (stepSize == 1){
            return input;
        }

        int outRows = (input.length - 1) * stepSize + 1;
        int outCols = (input[0].length - 1) * stepSize + 1;

        double[][] output = new double[outRows][outCols];

        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[0].length; j++) {
                output[i * stepSize][j * stepSize] = input[i][j];
            }
        }

        return output;
    }


    public void generateFilters(int n){
        List<double[][]> filters = new ArrayList<>();
        Random rd = new Random(SEED);

        for (int i = 0; i < n; i++) {

            double[][] filter = new double[this.filterSize][this.filterSize];

            for (int j = 0; j < this.filterSize; j++) {
                for (int k = 0; k < this.filterSize; k++) {
                    double value = rd.nextGaussian();
                    filter[j][k] = value;
                }
            }
            filters.add(filter);
        }
        this.filters = filters;
    }


    @Override
    public double[] getOutput(List<double[][]> input) {
        List<double[][]> output = convolutionalForwardPass(input);
        return nextLayer.getOutput(output);
    }

    @Override
    public double[] getOutput(double[] input) {
        List<double[][]> list = vectorToMatrix(input, this.inLenght, this.inRows, this.inCols);
        return getOutput(list);
    }

    @Override
    public void backPropagation(List<double[][]> diffLoss_diffOut) {

        List<double[][]> filtersDelta = new ArrayList<>();
        List<double[][]> diffLoss_diffOut_previousLayer = new ArrayList<>();

        for (int i = 0; i < filters.size(); i++) {
            filtersDelta.add(new double[filterSize][filterSize]);
        }

        for (int i = 0; i < lastInput.size(); i++) {

            double[][] errorInput = new double[inRows][inCols];

            for (int j = 0; j < filters.size(); j++) {

                double[][] currentFilter = filters.get(j);

                double[][] error = diffLoss_diffOut.get(i * filters.size() + j);
                double[][] spacedError = spaceArray(error);

                double[][] diffLoss_diffF = convolve(lastInput.get(i), spacedError, 1);

                double[][] delta = multiply(diffLoss_diffF, learningRate * -1);

                double[][] newTotalDelta = add(filtersDelta.get(j), delta);
                filtersDelta.set(j, newTotalDelta);

                double[][] flippedError = flipArrayHorizontally(flipArrayVertically(spacedError));
                errorInput = add(errorInput, fullConvolution(currentFilter, flippedError));
            }

            diffLoss_diffOut_previousLayer.add(errorInput);
        }

        for (int i = 0; i < filters.size(); i++) {
            double[][] modified = add(filtersDelta.get(i), filters.get(i));
            filters.set(i, modified);

        }

        if(previousLayer != null){
            previousLayer.backPropagation(diffLoss_diffOut_previousLayer);
        }
    }

    public double[][] flipArrayHorizontally(double[][] array){
        int rows = array.length;
        int cols = array[0].length;
        double[][] output = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                output[i][j] = array[i][j];
            }
        }

        return output;
    }

    @Override
    public void setNextLayer(Layer layer) {
        super.setNextLayer(layer);
    }

    @Override
    public void backPropagation(double[] diffLoss_diffOut) {
        List<double[][]> vector = this.vectorToMatrix(diffLoss_diffOut, inLenght, inRows, inCols);
        backPropagation(vector);

    }

    public double[][] flipArrayVertically(double[][] array){
        int rows = array.length;
        int cols = array[0].length;
        double[][] output = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                output[i][cols - j - 1] = array[i][j];
            }
        }

        return output;
    }


    @Override
    public int getOutputLength() {
        return filters.size() * inLenght;
    }

    @Override
    public int getOutputRows() {
        return (inRows - filterSize) / stepSize + 1;
    }

    @Override
    public int getOutputCols() {
        return (inCols - filterSize) / stepSize + 1;
    }

    @Override
    public int getOutputElements() {
        return getOutputCols() * getOutputRows() * getOutputLength();
    }

    @Override
    public double[] matrixToVector(List<double[][]> input) {
        return super.matrixToVector(input);
    }

    @Override
    public List<double[][]> vectorToMatrix(double[] input, int lenght, int rows, int cols) {
        return super.vectorToMatrix(input, lenght, rows, cols);
    }
}
