package de.jrk.neuralnetwork.training;

import java.util.ArrayList;

import de.jrk.neuralnetwork.Matrix;
import de.jrk.neuralnetwork.NeuralNetwork;

public class BackpropagationTrainer {
	private final NeuralNetwork nn;
	private ArrayList<Matrix[]> trainingData = new ArrayList<Matrix[]>();
	private ArrayList<Matrix[]> validationData = new ArrayList<Matrix[]>();
	private double learningRate;
	private double loss;
	private double validationLoss;

	public void train() {
		double loss = 0;
		for (Matrix[] data : trainingData) {
			Matrix[] errors = new Matrix[nn.getWeights().length];
			Matrix outputs = nn.feedforward(data[0]);
			Matrix outputErrors = data[1].subtract(outputs);
			errors[errors.length - 1] = outputErrors;
			for (int i = errors.length - 2; i >= 0; i--) {
				errors[i] = nn.getWeights()[i + 1].transpose().multiply(errors[i + 1]);
			}
			for (int w = 0; w < nn.getWeights().length; w++) {
				Matrix delta = errors[w].scale(learningRate);
				delta = delta.multiplyEntrywise(
						(nn.getActivationFunction().isDerivativeWithFunction() ? nn.getActivations()[w]
								: nn.getWeights()[w].multiply(w == 0 ? data[0] : nn.getActivations()[w - 1])
										.add(nn.getBiases()[w]))
												.map((x, i, j) -> nn.getActivationFunction().derivative(x)));
				Matrix deltaWeights = delta.multiply((w == 0 ? data[0] : nn.getActivations()[w - 1]).transpose());
				nn.getWeights()[w] = nn.getWeights()[w].add(deltaWeights);
				nn.getBiases()[w] = nn.getBiases()[w].add(delta);
			}
			for (int i = 0; i < outputErrors.getRows(); i++) {
				for (int j = 0; j < outputErrors.getCols(); j++) {
					loss += Math.abs(outputErrors.get(i, j));
				}
			}
		}
		this.loss = loss / trainingData.size();
		double validationLoss = 0;
		for (Matrix[] data : validationData) {
			Matrix[] errors = new Matrix[nn.getWeights().length];
			Matrix outputs = nn.feedforward(data[0]);
			Matrix outputErrors = data[1].subtract(outputs);
			errors[errors.length - 1] = outputErrors;

			for (int i = 0; i < outputErrors.getRows(); i++) {
				for (int j = 0; j < outputErrors.getCols(); j++) {
					validationLoss += Math.pow(outputErrors.get(i, j), 2);
				}
			}
		}
		this.validationLoss = validationLoss / validationData.size();
	}

	public BackpropagationTrainer(NeuralNetwork nn) {
		this.nn = nn;
	}

	public void setValidationData(ArrayList<Matrix[]> validationData) {
		this.validationData = validationData;
	}

	public ArrayList<Matrix[]> getValidationData() {
		return validationData;
	}

	public void setTrainingData(ArrayList<Matrix[]> trainingData) {
		this.trainingData = trainingData;
	}

	public ArrayList<Matrix[]> getTrainingData() {
		return trainingData;
	}

	public double getLoss() {
		return loss;
	}

	public double getValidationLoss() {
		return validationLoss;
	}

	public void setLearningRate(double learningRate) {
		this.learningRate = learningRate;
	}

	public double getLearningRate() {
		return learningRate;
	}

	public void addTrainingData(Matrix[]... trainingData) {
		for (int i = 0; i < trainingData.length; i++) {
			this.trainingData.add(trainingData[i]);
		}
	}

	public void addValidationData(Matrix[]... validationData) {
		for (int i = 0; i < validationData.length; i++) {
			this.validationData.add(validationData[i]);
		}
	}
}
