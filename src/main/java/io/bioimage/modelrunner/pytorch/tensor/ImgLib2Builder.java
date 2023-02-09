
package io.bioimage.modelrunner.pytorch.tensor;

import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.utils.IndexingUtils;

import net.imglib2.Cursor;
import net.imglib2.img.Img;
import net.imglib2.img.ImgFactory;
import net.imglib2.img.cell.CellImgFactory;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.DataType;

public class ImgLib2Builder {

	/**
	 * Creates a {@link Img} from a given {@link Tensor} and an array with its
	 * dimensions order.
	 * 
	 * @param tensor The tensor data is read from.
	 * @return The INDArray built from the tensor.
	 * @throws IllegalArgumentException If the tensor type is not supported.
	 */
	public static <T extends Type<T>> Img<T> build(NDArray tensor)
		throws IllegalArgumentException
	{
		// Create an Icy sequence of the same type of the tensor
		switch (tensor.getDataType()) {
			case UINT8:
			case INT8:
				return (Img<T>) buildFromTensorByte(tensor);
			case INT32:
				return (Img<T>) buildFromTensorInt(tensor);
			case FLOAT32:
				return (Img<T>) buildFromTensorFloat(tensor);
			case FLOAT64:
				return (Img<T>) buildFromTensorDouble(tensor);
			default:
				throw new IllegalArgumentException("Unsupported tensor type: " + tensor
					.getDataType());
		}
	}

	/**
	 * Builds a {@link Img} from a unsigned byte-typed {@link NDArray}.
	 * 
	 * @param tensor The tensor data is read from.
	 * @return The Img built from the tensor of type {@link DataType#UINT8}.
	 */
	private static Img<ByteType> buildFromTensorByte(NDArray tensor) {
		long[] tensorShape = tensor.getShape().getShape();
		final ImgFactory<ByteType> factory = new CellImgFactory<>(new ByteType(),
			5);
		final Img<ByteType> outputImg = factory.create(tensorShape);
		Cursor<ByteType> tensorCursor = outputImg.cursor();
		byte[] flatArr = tensor.toByteArray();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
			int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos,
				tensorShape);
			byte val = flatArr[flatPos];
			tensorCursor.get().set(val);
		}
		return outputImg;
	}

	/**
	 * Builds a {@link Img} from a unsigned integer-typed {@link NDArray}.
	 * 
	 * @param tensor The tensor data is read from.
	 * @return The INDArray built from the tensor of type {@link DataType#INT32}.
	 */
	private static Img<IntType> buildFromTensorInt(NDArray tensor) {
		long[] tensorShape = tensor.getShape().getShape();
		final ImgFactory<IntType> factory = new CellImgFactory<>(new IntType(), 5);
		final Img<IntType> outputImg = factory.create(tensorShape);
		Cursor<IntType> tensorCursor = outputImg.cursor();
		int[] flatArr = tensor.toIntArray();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
			int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos,
				tensorShape);
			int val = flatArr[flatPos];
			tensorCursor.get().set(val);
		}
		return outputImg;
	}

	/**
	 * Builds a {@link Img} from a unsigned float-typed {@link Tensor}.
	 * 
	 * @param tensor The tensor data is read from.
	 * @return The INDArray built from the tensor of type
	 *         {@link DataType#FLOAT32}.
	 */
	private static Img<FloatType> buildFromTensorFloat(NDArray tensor) {
		long[] tensorShape = tensor.getShape().getShape();
		final ImgFactory<FloatType> factory = new CellImgFactory<>(new FloatType(),
			5);
		final Img<FloatType> outputImg = factory.create(tensorShape);
		Cursor<FloatType> tensorCursor = outputImg.cursor();
		float[] flatArr = tensor.toFloatArray();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
			int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos,
				tensorShape);
			float val = flatArr[flatPos];
			tensorCursor.get().set(val);
		}
		return outputImg;
	}

	/**
	 * Builds a {@link Img} from a unsigned double-typed {@link NDArray}.
	 * 
	 * @param tensor The tensor data is read from.
	 * @return The INDArray built from the tensor of type
	 *         {@link DataType#FLOAT64}.
	 */
	private static Img<DoubleType> buildFromTensorDouble(NDArray tensor) {
		long[] tensorShape = tensor.getShape().getShape();
		final ImgFactory<DoubleType> factory = new CellImgFactory<>(
			new DoubleType(), 5);
		final Img<DoubleType> outputImg = factory.create(tensorShape);
		Cursor<DoubleType> tensorCursor = outputImg.cursor();
		double[] flatArr = tensor.toDoubleArray();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
			int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos,
				tensorShape);
			double val = flatArr[flatPos];
			tensorCursor.get().set(val);
		}
		return outputImg;
	}
}
