package de.jrk.neuralnetwork.tests;

import static org.junit.Assert.*;

import org.junit.Test;
import de.jrk.neuralnetwork.Matrix;

public class MatrixTest {

	@Test
	public void testConstructor() {
		Matrix matrix = new Matrix(3, 2);
		assertEquals(3, matrix.getRows());
		assertEquals(2, matrix.getCols());
	}

	@Test
	public void testFromString() {
		String string = "[[0.43, 0.25], [0.67, 0.23], [0.87, 0.85]]";
		Matrix matrix = Matrix.fromString(string);
		assertEquals(0.43, matrix.get(0, 0), 0);
		assertEquals(0.25, matrix.get(0, 1), 0);
		assertEquals(0.67, matrix.get(1, 0), 0);
		assertEquals(0.23, matrix.get(1, 1), 0);
		assertEquals(0.87, matrix.get(2, 0), 0);
		assertEquals(0.85, matrix.get(2, 1), 0);
	}

	@Test
	public void testToString() {
		String string = "[[0.43, 0.25], [0.67, 0.23], [0.87, 0.85]]";
		assertEquals(Matrix.fromString(string).toString(), string);
	}

	@Test
	public void testAddMatrix() {
		Matrix m1 = Matrix.fromString("[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]");
		Matrix m2 = Matrix.fromString("[[0.4, 0.1], [0.2, 0.8], [0.3, 0.7]]");
		Matrix result = Matrix.fromString("[[0.5, 0.3], [0.5, 1.2], [0.8, 1.3]]");
		assertEquals(result, m1.add(m2).map((x, i, j) -> Math.round(x * 10) / 10d));
	}

	@Test
	public void testSubtract() {
		Matrix m1 = Matrix.fromString("[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]");
		Matrix m2 = Matrix.fromString("[[0.4, 0.1], [0.2, 0.8], [0.3, 0.7]]");
		Matrix result = Matrix.fromString("[[-0.3, 0.1], [0.1, -0.4], [0.2, -0.1]]");
		assertEquals(result, m1.subtract(m2).map((x, i, j) -> Math.round(x * 10) / 10d));
	}

	@Test
	public void testAddNumber() {
		Matrix m = Matrix.fromString("[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]");
		Matrix result = Matrix.fromString("[[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]");
		assertEquals(result, m.add(0.4).map((x, i, j) -> Math.round(x * 10) / 10d));
	}

	@Test
	public void testScale() {
		Matrix m = Matrix.fromString("[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]");
		Matrix result = Matrix.fromString("[[0.04, 0.08], [0.12, 0.16], [0.2, 0.24]]");
		assertEquals(result, m.scale(0.4).map((x, i, j) -> Math.round(x * 100) / 100d));
	}

	@Test
	public void testTranspose() {
		assertEquals(Matrix.fromString("[[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]"),
				Matrix.fromString("[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]").transpose());
	}

	@Test
	public void testMultiply() {
		Matrix m1 = Matrix.fromString("[[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]");
		Matrix m2 = Matrix.fromString("[[0.4, 0.1], [0.2, 0.8], [0.3, 0.7]]");
		Matrix result = Matrix.fromString("[[0.25, 0.6], [0.34, 0.76]]");
		assertEquals(result, m1.multiply(m2).map((x, i, j) -> Math.round(x * 100) / 100d));
	}

	@Test
	public void testMultiplyEntrywise() {
		Matrix m1 = Matrix.fromString("[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]");
		Matrix m2 = Matrix.fromString("[[0.4, 0.1], [0.2, 0.8], [0.3, 0.7]]");
		Matrix result = Matrix.fromString("[[0.04, 0.02], [0.06, 0.32], [0.15, 0.42]]");
		assertEquals(result, m1.multiplyEntrywise(m2).map((x, i, j) -> Math.round(x * 100) / 100d));
	}

	@Test
	public void testMap() {
		String string = "[[0.43, 0.25], [0.67, 0.23], [0.87, 0.85]]";
		Matrix matrix = Matrix.fromString(string).map((x, i, j) -> Math.round((1 - x) * Math.pow(i + 1, j) * 100) / 100d);
		assertEquals(0.57, matrix.get(0, 0), 0);
		assertEquals(0.75, matrix.get(0, 1), 0);
		assertEquals(0.33, matrix.get(1, 0), 0);
		assertEquals(1.54, matrix.get(1, 1), 0);
		assertEquals(0.13, matrix.get(2, 0), 0);
		assertEquals(0.45, matrix.get(2, 1), 0);
	}
	
	@Test
	public void testGetCopy() {
		Matrix m = Matrix.fromString("[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]");
		assertEquals(m, m.getCopy());
	}
	
	@Test
	public void testFrom2DArray() {
		assertEquals(Matrix.fromString("[[4], [6], [2], [5]]"), Matrix.from2DArray(4, 6, 2, 5));
	}
}
