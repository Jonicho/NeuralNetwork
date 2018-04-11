package de.jrk.neuralnetwork.tests;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

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
        nn.getActivationFunctions()[2] = ActivationFunction
                .getActivationFunction(ActivationFunction.INVERSE_SQUARE_ROOT_UNIT);
        nn.getActivationFunctions()[3] = ActivationFunction
                .getActivationFunction(ActivationFunction.RECTIFIED_LINEAR_UNIT);
        nn.randomize(1);
        nn.feedforward(Matrix.from2DArray(7, 4, 2, 5, 8, 4));
        NeuralNetwork nnCopy = nn.getCopy();
        assertTrue(Arrays.equals(nn.getWeights(), nnCopy.getWeights()));
        assertTrue(Arrays.equals(nn.getBiases(), nnCopy.getBiases()));
        assertTrue(Arrays.equals(nn.getActivations(), nnCopy.getActivations()));
        assertTrue(Arrays.equals(nn.getNetInputs(), nnCopy.getNetInputs()));
        assertTrue(Arrays.equals(nn.getActivationFunctions(), nnCopy.getActivationFunctions()));
        assertEquals(nn.feedforward(Matrix.from2DArray(7, 4, 2, 5, 8, 4)),
                nnCopy.feedforward(Matrix.from2DArray(7, 4, 2, 5, 8, 4)));
    }

    @Test
    public void toStringFromStringTest() {
        NeuralNetwork nn = new NeuralNetwork(6, 3, 6, 2, 1);
        nn.getActivationFunctions()[0] = ActivationFunction.getActivationFunction(ActivationFunction.GAUSSIAN);
        nn.getActivationFunctions()[1] = ActivationFunction.getActivationFunction(ActivationFunction.IDENTITY);
        nn.getActivationFunctions()[2] = ActivationFunction
                .getActivationFunction(ActivationFunction.INVERSE_SQUARE_ROOT_UNIT);
        nn.getActivationFunctions()[3] = ActivationFunction
                .getActivationFunction(ActivationFunction.RECTIFIED_LINEAR_UNIT);
        nn.randomize(1);
        nn.feedforward(Matrix.from2DArray(7, 4, 2, 5, 8, 4));
        NeuralNetwork nnCopy = NeuralNetwork.fromString(nn.toString());
        assertTrue(Arrays.equals(nn.getWeights(), nnCopy.getWeights()));
        assertTrue(Arrays.equals(nn.getBiases(), nnCopy.getBiases()));
        assertTrue(Arrays.equals(nn.getActivationFunctions(), nnCopy.getActivationFunctions()));
        assertEquals(nn.feedforward(Matrix.from2DArray(7, 4, 2, 5, 8, 4)),
                nnCopy.feedforward(Matrix.from2DArray(7, 4, 2, 5, 8, 4)));
    }

    @Test
    public void feedforwardTest() {
        NeuralNetwork nn = NeuralNetwork.fromString(
                "{identity:[[-0.38200963299476, -0.44668023600730655, -0.5325481453106227], [-0.14955743530781218, -0.032385366731806364, 0.35515845828856674], [0.22687706523715812, 0.06290575015235333, 0.9181215299901346], [0.671950966469588, 0.22016013134670787, -0.24817526330107698]];[[-0.7426839465845265], [0.8790334421350359], [-0.7323018994344781], [0.17944180966571177]]}{rectified_linear_unit:[[-0.5135570805374232, 0.6956409944839117, 0.5519519116974321, 0.9732396339273677], [0.28133586671360544, 0.023361905799160088, -0.027463131764554838, -0.6830377602653557]];[[-0.5621854938254964], [0.3033999975340649]]}{softsign_norm:[[0.1256055731128023, 0.880811443572094], [-0.27942119375567986, -0.3048032845115842], [0.2066561870761079, 0.6810660379168623], [0.6227420318344354, 0.1074797184122267], [0.6300406880109002, -0.05586477479047791], [-0.05646511989613523, -0.3461794869513206]];[[-0.03491292359067821], [0.7473244990629018], [0.3046019034550522], [0.4582992966721533], [-0.09210909837148007], [-0.2816423845692142]]}");
        assertEquals(Matrix.fromString(
                "[[0.7922501351170295], [0.1445184236796418], [0.8640019615483391], [0.9419003058392763], [0.9385693547749645], [0.25910391161218943]]"),
                nn.feedforward(Matrix.from2DArray(7.4, 6.8, 2.2)));

    }
}