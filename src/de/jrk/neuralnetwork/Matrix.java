package de.jrk.neuralnetwork;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Matrix {
	private final double[][] data;

	public Matrix(int rows, int cols) {
		data = new double[rows][cols];
	}

	public double get(int i, int j) {
		return data[i][j];
	}

	public void set(int i, int j, double x) {
		data[i][j] = x;
	}

	public int getRows() {
		return data.length;
	}

	public int getCols() {
		return data[0].length;
	}

	public Matrix getCopy() {
		return (Matrix) clone();
	}

	@Override
	protected Object clone() {
		Matrix result = new Matrix(getRows(), getCols());
		for (int i = 0; i < getRows(); i++) {
			for (int j = 0; j < getCols(); j++) {
				result.set(i, j, get(i, j));
			}
		}
		return result;
	}

	@Override
	public String toString() {
		return Arrays.deepToString(data);
	}

	public Matrix add(Matrix m) {
		if (getRows() != m.getRows() || getCols() != m.getCols()) {
			throw new IllegalArgumentException("The size of this Matrix does not match the size of the given Matrix!");
		}
		Matrix result = new Matrix(getRows(), getCols());
		for (int i = 0; i < result.getRows(); i++) {
			for (int j = 0; j < result.getCols(); j++) {
				result.set(i, j, get(i, j) + m.get(i, j));
			}
		}
		return result;
	}

	public Matrix subtract(Matrix m) {
		if (getRows() != m.getRows() || getCols() != m.getCols()) {
			throw new IllegalArgumentException("The size of this Matrix does not match the size of the given Matrix!");
		}
		Matrix result = new Matrix(getRows(), getCols());
		for (int i = 0; i < result.getRows(); i++) {
			for (int j = 0; j < result.getCols(); j++) {
				result.set(i, j, get(i, j) - m.get(i, j));
			}
		}
		return result;
	}

	public Matrix add(double x) {
		Matrix result = new Matrix(getRows(), getCols());
		for (int i = 0; i < result.getRows(); i++) {
			for (int j = 0; j < result.getCols(); j++) {
				result.set(i, j, get(i, j) + x);
			}
		}
		return result;
	}

	public Matrix scale(double scalar) {
		Matrix result = new Matrix(getRows(), getCols());
		for (int i = 0; i < result.getRows(); i++) {
			for (int j = 0; j < result.getCols(); j++) {
				result.set(i, j, get(i, j) * scalar);
			}
		}
		return result;
	}

	public Matrix transpose() {
		Matrix result = new Matrix(getCols(), getRows());
		for (int i = 0; i < result.getRows(); i++) {
			for (int j = 0; j < result.getCols(); j++) {
				result.set(i, j, get(j, i));
			}
		}
		return result;
	}

	public Matrix multiply(Matrix m) {
		if (getCols() != m.getRows()) {
			throw new IllegalArgumentException("The columns of this Matrix do not match the rows of the given Matrix!");
		}
		Matrix result = new Matrix(getRows(), m.getCols());
		for (int i = 0; i < result.getRows(); i++) {
			for (int j = 0; j < result.getCols(); j++) {
				double sum = 0;
				for (int k = 0; k < getCols(); k++) {
					sum += get(i, k) * m.get(k, j);
				}
				result.set(i, j, sum);
			}
		}
		return result;
	}

	public Matrix multiplyEntrywise(Matrix m) {
		if (getRows() != m.getRows() || getCols() != m.getCols()) {
			throw new IllegalArgumentException("The size of this Matrix does not match the size of the given Matrix!");
		}
		Matrix result = new Matrix(getRows(), getCols());
		for (int i = 0; i < result.getRows(); i++) {
			for (int j = 0; j < result.getCols(); j++) {
				result.set(i, j, get(i, j) * m.get(i, j));
			}
		}
		return result;
	}

	public static Matrix from2DArray(double... array) {
		Matrix result = new Matrix(array.length, 1);
		for (int i = 0; i < array.length; i++) {
			result.set(i, 0, array[i]);
		}
		return result;
	}

	public static Matrix fromString(String string) {
		Pattern pattern = Pattern.compile("\\[([^\\[\\]]*)\\]");
		Matcher matcher = pattern.matcher(string);
		ArrayList<double[]> g = new ArrayList<double[]>();
		int cols = 0;
		while (matcher.find()) {
			String[] s = matcher.group(1).split(", ");
			if (g.size() == 0) {
				cols = s.length;
			} else if (s.length != cols) {
				throw new IllegalArgumentException("Invalid matrix string!");
			}
			double[] d = new double[s.length];
			for (int i = 0; i < s.length; i++) {
				d[i] = Double.parseDouble(s[i]);
			}
			g.add(d);
		}
		Matrix result = new Matrix(g.size(), cols);
		for (int i = 0; i < result.getRows(); i++) {
			for (int j = 0; j < result.getCols(); j++) {
				result.set(i, j, g.get(i)[j]);
			}
		}
		return result;
	}
}
