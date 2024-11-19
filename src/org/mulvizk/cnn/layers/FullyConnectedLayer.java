package org.mulvizk.cnn.layers;

import java.util.*;

public class FullyConnectedLayer extends Layer {

    private long SEED;
    private double[][] weights;
    private int inLenght;
    private int outLenght;
    private double learningRate;
    public double[] lastZ;
    public double[] lastInput;

    public FullyConnectedLayer(int inLenght, int outLenght, double learningRate, long SEED) {
        this.inLenght = inLenght;
        this.outLenght = outLenght;
        this.learningRate = learningRate;
        this.SEED = SEED;
        weights = new double[inLenght][outLenght];
        this.generateWeights();
    }

    public FullyConnectedLayer(int inLenght, int outLenght, double learningRate, double[][] weights) {
        this.inLenght = inLenght;
        this.outLenght = outLenght;
        this.learningRate = learningRate;
        this.weights = weights;
    }

    public double [] fullyConnectedForwardPass(double[] input){

        this.lastInput = input;
        double[] z = new double[outLenght];
        double[] output = new double[outLenght];

        for (int i = 0; i < this.inLenght; i++) {
            for (int j = 0; j < this.outLenght; j++) {
                z[j] += input[i] * this.weights[i][j];
            }
        }

        this.lastZ = z;

        for (int i = 0; i < this.inLenght; i++) {
            for (int j = 0; j < this.outLenght; j++) {
                output[j] = relu(z[j]);
            }
        }

        return output;
    }


    @Override
    public double[] getOutput(List<double[][]> input) {
        double[] vector = matrixToVector(input);
        return this.getOutput(vector);
    }

    @Override
    public double[] getOutput(double[] input) {
        double[] forwardPass = this.fullyConnectedForwardPass(input);
        return (this.nextLayer != null) ? this.nextLayer.getOutput(forwardPass) : forwardPass;
    }

    @Override
    public void backPropagation(List<double[][]> diffLoss_diffOut) {
        double[] vector = this.matrixToVector(diffLoss_diffOut);
        backPropagation(vector);

    }

    @Override
    public void backPropagation(double[] diffLoss_diffOut) {

        double[] diffLoss_diffX = new double[inLenght];
        double diffOut_diffZ;
        double diffZ_diffW;
        double diffLoss_diffW;
        double diffZ_diffX;

        for (int i = 0; i < inLenght; i++) {

            double diffLoss_diffIn_sum = 0;
            for (int j = 0; j < outLenght; j++) {

                diffOut_diffZ = diffRelu(lastZ[j]);
                diffZ_diffW = lastInput[i];
                diffZ_diffX = weights[i][j];

                diffLoss_diffW = diffLoss_diffOut[j] * diffOut_diffZ * diffZ_diffW;
                weights[i][j] -= diffLoss_diffW * learningRate;
                diffLoss_diffIn_sum += diffLoss_diffOut[j] * diffZ_diffX;
            }
            diffLoss_diffX[i] = diffLoss_diffIn_sum;
        }

        if(previousLayer != null){
            previousLayer.backPropagation(diffLoss_diffX);
        }

    }

    @Override
    public int getOutputLength() {
        return outLenght;
    }

    @Override
    public int getOutputRows() {
        return outLenght;
    }

    @Override
    public int getOutputCols() {
        return outLenght;
    }

    @Override
    public int getOutputElements() {
        return outLenght;
    }

    @Override
    public double[] matrixToVector(List<double[][]> input) {
        return super.matrixToVector(input);
    }

    @Override
    public List<double[][]> vectorToMatrix(double[] input, int lenght, int rows, int cols) {
        return super.vectorToMatrix(input, lenght, rows, cols);
    }

    public double relu(double input){
        return (input <= 0) ? 0 : input;
    }

    public double diffRelu(double input){
        return (input <= 0) ? 0 : 1;
    }

    public void generateWeights(){
        Random rd =  new Random(SEED);
        for (int i = 0; i < inLenght; i++) {
            for (int j = 0; j < outLenght; j++) {
                weights[i][j] = rd.nextGaussian();
            }
        }
    }

    public double[][] getWeights() {
        return weights;
    }

    public void setWeights(double[][] weights) {
        this.weights = weights;
    }
}
