package de.jrk.neuralnetwork;

public class NeuralNetworkTrainer {
	private final NeuralNetwork nn;

	public NeuralNetworkTrainer(NeuralNetwork nn) {
		this.nn = nn;
	}

	public double train(Matrix inputs, Matrix targets, double learningRate) {
		Matrix[] errors = new Matrix[nn.getWeights().length];
		Matrix outputs = nn.feedfoward(inputs);
		Matrix outputErrors = Matrix.subtract(targets, outputs);
		errors[errors.length - 1] = outputErrors;
		for (int i = errors.length - 2; i >= 0; i--) {
			errors[i] = Matrix.multiply(Matrix.transpose(nn.getWeights()[i + 1]), errors[i + 1]);
		}
		for (int w = 0; w < nn.getWeights().length; w++) {
			Matrix delta = Matrix.scale(errors[w], learningRate);
			delta = Matrix.multiplyEntrywise(delta,
					Matrix.multiplyEntrywise(nn.getActivations()[w], Matrix.scale(Matrix.add(nn.getActivations()[w], -1), -1)));

			Matrix deltaWeights = Matrix.multiply(delta,
					Matrix.transpose(w == 0 ? inputs : nn.getActivations()[w - 1]));
			nn.getWeights()[w] = Matrix.add(nn.getWeights()[w], deltaWeights);

			Matrix deltaBiases = Matrix.multiply(delta, Matrix.from2DArray(1));
			nn.getBiases()[w] = Matrix.add(nn.getBiases()[w], deltaBiases);
		}
		double loss = 0;
		for (int i = 0; i < outputErrors.getRows(); i++) {
			for (int j = 0; j < outputErrors.getCols(); j++) {
				loss += Math.pow(outputErrors.get(i, j), 2);
			}
		}
		return loss;
	}
}
