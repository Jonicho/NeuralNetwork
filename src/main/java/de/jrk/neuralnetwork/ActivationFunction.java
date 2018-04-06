package de.jrk.neuralnetwork;

public abstract class ActivationFunction {
	private final String name;
	private final boolean derivativeWithFunction;

	private ActivationFunction(String name, boolean derivativeWithFunction) {
		this.name = name;
		this.derivativeWithFunction = derivativeWithFunction;
	}

	public abstract double function(double x);

	public double derivative(double x) {
		return Double.NaN;
	}

	public String getName() {
		return name;
	}

	public boolean isDerivativeWithFunction() {
		return derivativeWithFunction;
	}

	@Override
	public String toString() {
		return "Activation function \"" + name + "\"";
	}

	public static final String IDENTITY = "identity", BINARY_STEP = "binary_step", SIGMOID = "sigmoid", TANH = "tanh",
			ARCTAN = "arctan", SOFTSIGN = "softsign", SOFTSIGN_NORM = "softsign_norm",
			INVERSE_SQUARE_ROOT_UNIT = "inverse_square_root_unit", RECTIFIED_LINEAR_UNIT = "rectified_linear_unit",
			SINUSOID = "sinusoid", SINC = "sinc", GAUSSIAN = "gaussian";

	public static ActivationFunction getActivationFunction(String function) {
		switch (function) {
		case IDENTITY:
			return new ActivationFunction(function, false) {
				@Override
				public double function(double x) {
					return x;
				}

				@Override
				public double derivative(double x) {
					return 1;
				}
			};
		case BINARY_STEP:
			return new ActivationFunction(function, false) {
				@Override
				public double function(double x) {
					return x < 0 ? 0 : 1;
				}

				@Override
				public double derivative(double x) {
					return x != 0 ? 0 : Double.NaN;
				}
			};
		case SIGMOID:
			return new ActivationFunction(function, true) {
				@Override
				public double function(double x) {
					return 1 / (1 + Math.exp(-x));
				}

				@Override
				public double derivative(double x) {
					return x * (1 - x);
				}
			};
		case TANH:
			return new ActivationFunction(function, true) {
				@Override
				public double function(double x) {
					return Math.tanh(x);
				}

				@Override
				public double derivative(double x) {
					return 1 - x * x;
				}
			};
		case ARCTAN:
			return new ActivationFunction(function, false) {
				@Override
				public double function(double x) {
					return Math.atan(x);
				}

				@Override
				public double derivative(double x) {
					return 1 / (x * x + 1);
				}
			};
		case SOFTSIGN:
			return new ActivationFunction(function, false) {
				@Override
				public double function(double x) {
					return x / (1 + Math.abs(x));
				}

				@Override
				public double derivative(double x) {
					return 1 / ((1 + Math.abs(x)) * (1 + Math.abs(x)));
				}
			};
		case SOFTSIGN_NORM:
			return new ActivationFunction(function, false) {
				@Override
				public double function(double x) {
					return 0.5 * x / (1 + Math.abs(x)) + 0.5;
				}

				@Override
				public double derivative(double x) {
					return 1 / (2 * (1 + Math.abs(x)) * (1 + Math.abs(x)));
				}
			};
		case INVERSE_SQUARE_ROOT_UNIT:
			return new ActivationFunction(function, false) {
				@Override
				public double function(double x) {
					return x / Math.sqrt(1 + x * x);
				}

				@Override
				public double derivative(double x) {
					return Math.pow(1 / Math.sqrt(1 + x * x), 3);
				}
			};
		case RECTIFIED_LINEAR_UNIT:
			return new ActivationFunction(function, false) {
				@Override
				public double function(double x) {
					return x < 0 ? 0 : x;
				}

				@Override
				public double derivative(double x) {
					return x < 0 ? 0 : 1;
				}
			};
		case SINUSOID:
			return new ActivationFunction(function, false) {
				@Override
				public double function(double x) {
					return Math.sin(x);
				}

				@Override
				public double derivative(double x) {
					return Math.cos(x);
				}
			};
		case SINC:
			return new ActivationFunction(function, false) {
				@Override
				public double function(double x) {
					return x == 0 ? 1 : Math.sin(x) / x;
				}

				@Override
				public double derivative(double x) {
					return x == 0 ? 0 : (Math.cos(x) / x) - (Math.sin(x) / (x * x));
				}
			};
		case GAUSSIAN:
			return new ActivationFunction(function, false) {
				@Override
				public double function(double x) {
					return Math.exp(-x * x);
				}

				@Override
				public double derivative(double x) {
					return -2 * x * Math.exp(-x * x);
				}
			};
		default:
			throw new IllegalArgumentException("Activation function \"" + function + "\" does not exist!");
		}
	}
}
