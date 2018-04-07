package de.jrk.neuralnetwork.tests;

import static org.junit.Assert.*;

import java.util.Arrays;

import org.junit.Test;

import de.jrk.neuralnetwork.ActivationFunction;
import de.jrk.neuralnetwork.Matrix;
import de.jrk.neuralnetwork.NeuralNetwork;

public class NeuralNetworkTest {

    @Test
    public void copyTest() {
        NeuralNetwork nn = new NeuralNetwork(6, 3, 6, 2, 1);
        nn.getActivationFunctions()[0] = ActivationFunction.getActivationFunction(ActivationFunction.GAUSSIAN);
        nn.getActivationFunctions()[1] = ActivationFunction.getActivationFunction(ActivationFunction.IDENTITY);
        nn.getActivationFunctions()[2] = ActivationFunction.getActivationFunction(ActivationFunction.INVERSE_SQUARE_ROOT_UNIT);
        nn.getActivationFunctions()[3] = ActivationFunction.getActivationFunction(ActivationFunction.RECTIFIED_LINEAR_UNIT);
        nn.randomize(1);
        nn.feedforward(Matrix.from2DArray(7,4,2,5,8,4));
        NeuralNetwork nnCopy = nn.getCopy();
        assertTrue(Arrays.equals(nn.getWeights(), nnCopy.getWeights()));
        assertTrue(Arrays.equals(nn.getBiases(), nnCopy.getBiases()));
        assertTrue(Arrays.equals(nn.getActivations(), nnCopy.getActivations()));
        assertTrue(Arrays.equals(nn.getNetInputs(), nnCopy.getNetInputs()));
        assertTrue(Arrays.equals(nn.getActivationFunctions(), nnCopy.getActivationFunctions()));
        assertEquals(nn.feedforward(Matrix.from2DArray(7,4,2,5,8,4)), nnCopy.feedforward(Matrix.from2DArray(7,4,2,5,8,4)));
    }

    @Test
    public void toStringFromStringTest() {
        NeuralNetwork nn = new NeuralNetwork(6, 3, 6, 2, 1);
        nn.getActivationFunctions()[0] = ActivationFunction.getActivationFunction(ActivationFunction.GAUSSIAN);
        nn.getActivationFunctions()[1] = ActivationFunction.getActivationFunction(ActivationFunction.IDENTITY);
        nn.getActivationFunctions()[2] = ActivationFunction.getActivationFunction(ActivationFunction.INVERSE_SQUARE_ROOT_UNIT);
        nn.getActivationFunctions()[3] = ActivationFunction.getActivationFunction(ActivationFunction.RECTIFIED_LINEAR_UNIT);
        nn.randomize(1);
        nn.feedforward(Matrix.from2DArray(7,4,2,5,8,4));
        NeuralNetwork nnCopy = NeuralNetwork.fromString(nn.toString());
        assertTrue(Arrays.equals(nn.getWeights(), nnCopy.getWeights()));
        assertTrue(Arrays.equals(nn.getBiases(), nnCopy.getBiases()));
        assertTrue(Arrays.equals(nn.getActivationFunctions(), nnCopy.getActivationFunctions()));
        assertEquals(nn.feedforward(Matrix.from2DArray(7,4,2,5,8,4)), nnCopy.feedforward(Matrix.from2DArray(7,4,2,5,8,4)));
    }

}