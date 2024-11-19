package org.mulvizk.cnn.network;

import org.mulvizk.cnn.layers.ConvolutionLayer;
import org.mulvizk.cnn.layers.FullyConnectedLayer;
import org.mulvizk.cnn.layers.Layer;
import org.mulvizk.cnn.layers.MaxPoolLayer;

import java.util.ArrayList;
import java.util.List;

public class NetworkBuilder {

    private NeuralNetwork nn;

    private int inRows;
    private int inCols;
    private double scaleFactor;
    List<Layer> layers;

    public NetworkBuilder(int inRows, int inCols, double scaleFactor) {
        this.inRows = inRows;
        this.inCols = inCols;
        this.scaleFactor = scaleFactor;
        layers = new ArrayList<>();
    }

    public void addConvolutionLayer(int n, int filterSize, int stepSize, double learningRate ,long SEED){
        if(layers.isEmpty()){
            layers.add(new ConvolutionLayer(filterSize, n, learningRate,  stepSize,1, inRows, inCols, SEED));
        } else {
            Layer prev = layers.get(layers.size() - 1);
            layers.add(new ConvolutionLayer(filterSize, n, learningRate, stepSize,
                    prev.getOutputLength(), prev.getOutputRows(), prev.getOutputCols(), SEED));
        }
    }

    public void addMaxPoolLayer(int windowSize, int stepSize){
        if(layers.isEmpty()){
            layers.add(new MaxPoolLayer(stepSize, windowSize, 1, inRows, inCols));
        } else {
            Layer prev = layers.get(layers.size() - 1);
            layers.add(new MaxPoolLayer(stepSize, windowSize, prev.getOutputLength(), prev.getOutputRows(), prev.getOutputCols()));
        }
    }

    public void addFullyConnectedLayer(int outLenght, double learningRate, long SEED){
        if(layers.isEmpty()){
            layers.add(new FullyConnectedLayer(inCols * inRows, outLenght, learningRate, SEED));
        } else {
            Layer prev = layers.get(layers.size() - 1);
            layers.add(new FullyConnectedLayer(prev.getOutputElements(), outLenght, learningRate, SEED));
        }
    }

    public NeuralNetwork build(){
        nn = new NeuralNetwork(layers, scaleFactor);
        return nn;
    }
}
