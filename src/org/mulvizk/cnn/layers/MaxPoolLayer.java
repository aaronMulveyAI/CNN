package org.mulvizk.cnn.layers;

import java.util.ArrayList;
import java.util.List;

public class MaxPoolLayer extends Layer{

    private int stepSize;
    private int windowSize;

    private int inLength;
    private int inRows;
    private int inCols;
    private List<int[][]> listMaxRow;
    private List<int[][]> listMaxCol;

    public MaxPoolLayer(int stepSize, int windowSize, int inLength, int inRows, int inCols) {
        this.stepSize = stepSize;
        this.windowSize = windowSize;
        this.inLength = inLength;
        this.inRows = inRows;
        this.inCols = inCols;
    }

    public List<double[][]> maxPoolForwardPass(List<double[][]> input){

        List<double[][]> output = new ArrayList<>();
        listMaxCol = new ArrayList<>();
        listMaxRow = new ArrayList<>();

        for (int i = 0; i < input.size(); i++) {
            output.add(pool(input.get(i)));
        }

        return output;
    }

    public double[][] pool(double[][] input){

        double[][] output = new double[this.getOutputRows()][this.getOutputCols()];
        int[][] maxRow = new int[this.getOutputRows()][this.getOutputCols()];
        int[][] maxCol = new int[this.getOutputRows()][this.getOutputCols()];

        for (int i = 0; i < getOutputRows(); i += stepSize) {
            for (int j = 0; j < getOutputCols(); j += stepSize) {

                double max = 0d;
                maxRow[i][j] = -1;
                maxCol[i][j] = -1;
                for (int k = 0; k < this.windowSize; k++) {
                    for (int l = 0; l < this.windowSize; l++) {
                        if(max < input[i + k][j + l]){
                            max = input[i + k][j + l];
                            maxRow[i][j] = i + k;
                            maxCol[i][j] = j + l;
                        }
                    }
                }
                output[i][j] = max;
            }
        }

        this.listMaxRow.add(maxRow);
        this.listMaxCol.add(maxCol);
        return output;
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        List<double[][]> output = maxPoolForwardPass(input);
        return nextLayer.getOutput(output);
    }

    @Override
    public double[] getOutput(double[] input) {
        List<double[][]> list = vectorToMatrix(input, this.inLength, this.inRows, this.inCols);
        return this.getOutput(list);
    }

    @Override
    public void backPropagation(List<double[][]> diffLoss_diffOut) {
        List<double[][]> diffX_diffLoss = new ArrayList<>();

        int l = 0;
        for (double[][] matrix : diffLoss_diffOut) {

            double[][] error = new double[inRows][inCols];
            for (int i = 0; i < this.getOutputRows(); i++) {
                for (int j = 0; j < this.getOutputCols(); j++) {

                    int max_i = this.listMaxRow.get(l)[i][j];
                    int max_j = this.listMaxCol.get(l)[i][j];

                    if(max_i != -1){
                        error[max_i][max_j] = matrix[i][j];
                    }
                }
            }

            diffX_diffLoss.add(error);
            l++;
        }

        if(this.previousLayer != null){
            previousLayer.backPropagation(diffX_diffLoss);
        }
    }

    @Override
    public void backPropagation(double[] diffLoss_diffOut) {
        List<double[][]> list = vectorToMatrix(
                diffLoss_diffOut,
                getOutputLength(),
                getOutputRows(),
                getOutputCols()
        );
        backPropagation(list);
    }

    @Override
    public int getOutputLength() {
        return inLength;
    }

    @Override
    public int getOutputRows() {
        return (inRows - windowSize) / stepSize + 1;
    }

    @Override
    public int getOutputCols() {
        return (inCols - windowSize) / stepSize + 1;
    }

    @Override
    public int getOutputElements() {
        return inLength * this.getOutputCols() * this.getOutputRows();
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
