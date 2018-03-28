package de.jrk.neuralnetwork;

import java.util.ArrayList;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class NeuralNetwork {
	private final Matrix[] weights;
	private final Matrix[] biases;
	private final Matrix[] activations;
	private final String activationFunction;
	
	public NeuralNetwork(int... neurons) {
		this(ActivationFunction.SOFTSIGN_NORM, neurons);
	}

	public NeuralNetwork(String activationFunction, int... neurons) {
		weights = new Matrix[neurons.length - 1];
		biases = new Matrix[neurons.length - 1];
		activations = new Matrix[neurons.length - 1];
		this.activationFunction = activationFunction;
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
			activations[a] = weights[a].multiply(a == 0 ? inputs : activations[a - 1]).add(biases[a])
					.map((x, i, j) -> ActivationFunction.function(activationFunction, x));
		}
		return activations[activations.length - 1].getCopy();
	}

	public Matrix[] getWeights() {
		return weights;
	}

	public Matrix[] getBiases() {
		return biases;
	}

	public Matrix[] getActivations() {
		return activations;
	}
	
	public String getActivationFunction() {
		return activationFunction;
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
		return nn;
	}

	@Override
	public String toString() {
		String result = activationFunction + ":";
		for (int i = 0; i < weights.length; i++) {
			result += "{" + weights[i] + ";" + biases[i] + "}";
		}
		return result;
	}

	public static NeuralNetwork fromString(String string) {
		String activationFunction = string.split(":")[0];
		string = string.split(":")[1];
		ArrayList<Matrix> weights = new ArrayList<Matrix>();
		ArrayList<Matrix> biases = new ArrayList<Matrix>();
		Pattern pattern = Pattern.compile("\\{([^\\{\\}]*)\\}");
		Matcher matcher = pattern.matcher(string);
		while (matcher.find()) {
			String[] layer = matcher.group(1).split(";");
			weights.add(Matrix.fromString(layer[0]));
			biases.add(Matrix.fromString(layer[1]));
		}
		int[] neurons = new int[weights.size() + 1];
		neurons[0] = weights.get(0).getCols();
		for (int i = 1; i < neurons.length; i++) {
			neurons[i] = weights.get(i - 1).getRows();
		}
		NeuralNetwork result = new NeuralNetwork(activationFunction, neurons);
		for (int i = 0; i < weights.size(); i++) {
			result.getWeights()[i] = weights.get(i);
			result.getBiases()[i] = biases.get(i);
		}
		return result;
	}
}
