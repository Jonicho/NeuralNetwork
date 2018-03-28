package de.jrk.neuralnetwork;

public class ActivationFunction {
	private ActivationFunction() {
	}

	public static final String IDENTITY = "identity", SIGMOID = "sigmoid", TANH = "tanh", SOFTSIGN = "softsign", SOFTSIGN_NORM = "softsign_norm";

	public static double function(String function, double x) {
		switch (function) {
		case IDENTITY:
			return x;
		case SIGMOID:
			return 1 / (1 + Math.exp(-x));
		case TANH:
			return Math.tanh(x);
		case SOFTSIGN:
			return x / (1 + Math.abs(x));
		case SOFTSIGN_NORM:
			return 0.5 * x / (1 + Math.abs(x)) + 0.5;
		default:
			throw new IllegalArgumentException("Activation function \"" + function + "\" does not exist!");
		}
	}
}
