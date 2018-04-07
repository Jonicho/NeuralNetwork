package de.jrk.neuralnetwork;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class NeuralNetwork {
	private final Matrix[] weights;
	private final Matrix[] biases;
	private final Matrix[] netInputs;
	private final Matrix[] activations;
	private final ActivationFunction[] activationFunctions;

	public NeuralNetwork(int... neurons) {
		this(ActivationFunction.getActivationFunction(ActivationFunction.SIGMOID), neurons);
	}

	public NeuralNetwork(ActivationFunction activationFunction, int... neurons) {
		weights = new Matrix[neurons.length - 1];
		biases = new Matrix[neurons.length - 1];
		netInputs = new Matrix[neurons.length - 1];
		activations = new Matrix[neurons.length - 1];
		activationFunctions = new ActivationFunction[neurons.length - 1];
		Arrays.fill(activationFunctions, activationFunction);
		for (int i = 1; i < neurons.length; i++) {
			weights[i - 1] = new Matrix(neurons[i], neurons[i - 1]);
			biases[i - 1] = new Matrix(neurons[i], 1);
		}
	}

	public void randomize(double range) {
		for (int l = 0; l < weights.length; l++) {
			weights[l] = weights[l].map((x, i, j) -> Math.random() * 2 * range - range);
			biases[l] = biases[l].map((x, i, j) -> Math.random() * 2 * range - range);
		}
	}

	public Matrix feedforward(Matrix inputs) {
		for (int a = 0; a < activations.length; a++) {
			netInputs[a] = weights[a].multiply(a == 0 ? inputs : activations[a - 1]).add(biases[a]);
			ActivationFunction activationFunction = activationFunctions[a];
			activations[a] = netInputs[a].map((x, i, j) -> activationFunction.function(x));
		}
		return activations[activations.length - 1].getCopy();
	}

	public Matrix[] getWeights() {
		return weights;
	}

	public Matrix[] getBiases() {
		return biases;
	}

	public Matrix[] getNetInputs() {
		return netInputs;
	}

	public Matrix[] getActivations() {
		return activations;
	}

	public ActivationFunction[] getActivationFunctions() {
		return activationFunctions;
	}

	public NeuralNetwork getCopy() {
		return (NeuralNetwork) clone();
	}

	@Override
	protected Object clone() {
		int[] neurons = new int[getWeights().length + 1];
		NeuralNetwork nn = new NeuralNetwork(neurons);
		for (int i = 0; i < getWeights().length; i++) {
			nn.getWeights()[i] = getWeights()[i].getCopy();
		}
		for (int i = 0; i < getBiases().length; i++) {
			nn.getBiases()[i] = getBiases()[i].getCopy();
		}
		for (int i = 0; i < getActivations().length; i++) {
			nn.getActivations()[i] = getActivations()[i] == null ? null : getActivations()[i].getCopy();
		}
		for (int i = 0; i < getNetInputs().length; i++) {
			nn.getNetInputs()[i] = getNetInputs()[i] == null ? null : getNetInputs()[i].getCopy();
		}
		for (int i = 0; i < getActivationFunctions().length; i++) {
			nn.getActivationFunctions()[i] = getActivationFunctions()[i];
		}
		return nn;
	}

	@Override
	public String toString() {
		String result = "";
		for (int i = 0; i < weights.length; i++) {
			result += "{" + activationFunctions[i].getName() + ":" + weights[i] + ";" + biases[i] + "}";
		}
		return result;
	}

	public static NeuralNetwork fromString(String string) {
		ArrayList<Matrix> weights = new ArrayList<Matrix>();
		ArrayList<Matrix> biases = new ArrayList<Matrix>();
		ArrayList<ActivationFunction> activationFunctions = new ArrayList<ActivationFunction>();
		Pattern pattern = Pattern.compile("\\{([^\\{\\}]*)\\}");
		Matcher matcher = pattern.matcher(string);
		while (matcher.find()) {
			String[] layer = matcher.group(1).split(":");
			String[] values = layer[1].split(";");
			weights.add(Matrix.fromString(values[0]));
			biases.add(Matrix.fromString(values[1]));
			activationFunctions.add(ActivationFunction.getActivationFunction(layer[0]));
		}
		int[] neurons = new int[weights.size() + 1];
		neurons[0] = weights.get(0).getCols();
		for (int i = 1; i < neurons.length; i++) {
			neurons[i] = weights.get(i - 1).getRows();
		}
		NeuralNetwork result = new NeuralNetwork(neurons);
		for (int i = 0; i < weights.size(); i++) {
			result.getWeights()[i] = weights.get(i);
			result.getBiases()[i] = biases.get(i);
			result.getActivationFunctions()[i] = activationFunctions.get(i);
		}
		return result;
	}
}
