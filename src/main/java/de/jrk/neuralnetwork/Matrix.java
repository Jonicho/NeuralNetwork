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

	public int getRows() {
		return data.length;
	}

	public int getCols() {
		return data[0].length;
	}

	public Matrix add(Matrix m) {
		if (getRows() != m.getRows() || getCols() != m.getCols()) {
			throw new IllegalArgumentException("The size of this Matrix does not match the size of the given Matrix!");
		}
		return map((x, i, j) -> x + m.get(i, j));
	}

	public Matrix subtract(Matrix m) {
		if (getRows() != m.getRows() || getCols() != m.getCols()) {
			throw new IllegalArgumentException("The size of this Matrix does not match the size of the given Matrix!");
		}
		return map((x, i, j) -> x - m.get(i, j));
	}

	public Matrix add(double addend) {
		return map((x, i, j) -> x + addend);
	}

	public Matrix scale(double scalar) {
		return map((x, i, j) -> x * scalar);
	}

	public Matrix transpose() {
		return new Matrix(getCols(), getRows()).map((x, i, j) -> get(j, i));
	}

	public Matrix multiply(Matrix m) {
		if (getCols() != m.getRows()) {
			throw new IllegalArgumentException("The columns of this Matrix do not match the rows of the given Matrix!");
		}
		return new Matrix(getRows(), m.getCols()).map((x, i, j) -> {
			double sum = 0;
			for (int k = 0; k < getCols(); k++) {
				sum += get(i, k) * m.get(k, j);
			}
			return sum;
		});
	}

	public Matrix multiplyEntrywise(Matrix m) {
		if (getRows() != m.getRows() || getCols() != m.getCols()) {
			throw new IllegalArgumentException("The size of this Matrix does not match the size of the given Matrix!");
		}
		return map((x, i, j) -> x * m.get(i, j));
	}

	public Matrix map(MapFunction function) {
		Matrix result = new Matrix(getRows(), getCols());
		for (int i = 0; i < result.getRows(); i++) {
			for (int j = 0; j < result.getCols(); j++) {
				result.data[i][j] = function.function(get(i, j), i, j);
			}
		}
		return result;
	}

	public interface MapFunction {
		double function(double x, int i, int j);
	}

	public Matrix getCopy() {
		return (Matrix) clone();
	}

	@Override
	protected Object clone() {
		return map((x, i, j) -> x);
	}

	@Override
	public String toString() {
		return Arrays.deepToString(data);
	}

	@Override
	public boolean equals(Object object) {
		if (object == null || !(object instanceof Matrix)) {
			return false;
		}
		Matrix matrix = (Matrix) object;
		if (getRows() != matrix.getRows() || getCols() != matrix.getCols()) {
			return false;
		}
		for (int i = 0; i < matrix.getRows(); i++) {
			for (int j = 0; j < matrix.getCols(); j++) {
				if (matrix.get(i, j) != get(i, j)) {
					return false;
				}
			}
		}
		return true;
	}

	public static Matrix from2DArray(double... array) {
		return new Matrix(array.length, 1).map((x, i, j) -> array[i]);
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
		return new Matrix(g.size(), cols).map((x, i, j) -> g.get(i)[j]);
	}
}