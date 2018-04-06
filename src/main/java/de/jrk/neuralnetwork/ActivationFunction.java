package de.jrk.neuralnetwork;

public class ActivationFunction {
	private ActivationFunction() {
	}

	public static final String IDENTITY = "identity", BINARY_STEP = "binary_step", SIGMOID = "sigmoid", TANH = "tanh",
			ARCTAN = "arctan", SOFTSIGN = "softsign", SOFTSIGN_NORM = "softsign_norm",
			INVERSE_SQUARE_ROOT_UNIT = "inverse_square_root_unit", RECTIFIED_LINEAR_UNIT = "rectified_linear_unit",
			SINUSOID = "sinusoid", SINC = "sinc", GAUSSIAN = "gaussian";

	public static double function(String function, double x) {
		switch (function) {
		case IDENTITY:
			return x;
		case BINARY_STEP:
			return x < 0 ? 0 : 1;
		case SIGMOID:
			return 1 / (1 + Math.exp(-x));
		case TANH:
			return Math.tanh(x);
		case ARCTAN:
			return Math.atan(x);
		case SOFTSIGN:
			return x / (1 + Math.abs(x));
		case SOFTSIGN_NORM:
			return 0.5 * x / (1 + Math.abs(x)) + 0.5;
		case INVERSE_SQUARE_ROOT_UNIT:
			return x / Math.sqrt(1 + x * x);
		case RECTIFIED_LINEAR_UNIT:
			return x < 0 ? 0 : x;
		case SINUSOID:
			return Math.sin(x);
		case SINC:
			return x == 0 ? 1 : Math.sin(x) / x;
		case GAUSSIAN:
			return Math.exp(-x * x);
		default:
			throw new IllegalArgumentException("Activation function \"" + function + "\" does not exist!");
		}
	}
}
