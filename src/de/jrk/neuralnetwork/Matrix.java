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

	public static Matrix add(Matrix m1, Matrix m2) {
		if (m1.getRows() != m2.getRows() || m1.getCols() != m2.getCols()) {
			throw new IllegalArgumentException("The size of Matrix 1 does not match the size of Matrrix 2!");
		}
		Matrix result = new Matrix(m1.getRows(), m1.getCols());
		for (int i = 0; i < result.getRows(); i++) {
			for (int j = 0; j < result.getCols(); j++) {
				result.set(i, j, m1.get(i, j) + m2.get(i, j));
			}
		}
		return result;
	}

	public static Matrix subtract(Matrix m1, Matrix m2) {
		if (m1.getRows() != m2.getRows() || m1.getCols() != m2.getCols()) {
			throw new IllegalArgumentException("The size of Matrix 1 does not match the size of Matrrix 2!");
		}
		Matrix result = new Matrix(m1.getRows(), m1.getCols());
		for (int i = 0; i < result.getRows(); i++) {
			for (int j = 0; j < result.getCols(); j++) {
				result.set(i, j, m1.get(i, j) - m2.get(i, j));
			}
		}
		return result;
	}

	public static Matrix add(Matrix m, double x) {
		Matrix result = new Matrix(m.getRows(), m.getCols());
		for (int i = 0; i < result.getRows(); i++) {
			for (int j = 0; j < result.getCols(); j++) {
				result.set(i, j, m.get(i, j) + x);
			}
		}
		return result;
	}

	public static Matrix scale(Matrix m, double scalar) {
		Matrix result = new Matrix(m.getRows(), m.getCols());
		for (int i = 0; i < result.getRows(); i++) {
			for (int j = 0; j < result.getCols(); j++) {
				result.set(i, j, m.get(i, j) * scalar);
			}
		}
		return result;
	}

	public static Matrix transpose(Matrix m) {
		Matrix result = new Matrix(m.getCols(), m.getRows());
		for (int i = 0; i < result.getRows(); i++) {
			for (int j = 0; j < result.getCols(); j++) {
				result.set(i, j, m.get(j, i));
			}
		}
		return result;
	}

	public static Matrix multiply(Matrix m1, Matrix m2) {
		if (m1.getCols() != m2.getRows()) {
			throw new IllegalArgumentException("The columns of Matrix 1 do not match the rows of Matrrix 2!");
		}
		Matrix result = new Matrix(m1.getRows(), m2.getCols());
		for (int i = 0; i < result.getRows(); i++) {
			for (int j = 0; j < result.getCols(); j++) {
				double sum = 0;
				for (int k = 0; k < m1.getCols(); k++) {
					sum += m1.get(i, k) * m2.get(k, j);
				}
				result.set(i, j, sum);
			}
		}
		return result;
	}

	public static Matrix multiplyEntrywise(Matrix m1, Matrix m2) {
		if (m1.getRows() != m2.getRows() || m1.getCols() != m2.getCols()) {
			throw new IllegalArgumentException("The size of Matrix 1 does not match the size of Matrrix 2!");
		}
		Matrix result = new Matrix(m1.getRows(), m1.getCols());
		for (int i = 0; i < result.getRows(); i++) {
			for (int j = 0; j < result.getCols(); j++) {
				result.set(i, j, m1.get(i, j) * m2.get(i, j));
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
