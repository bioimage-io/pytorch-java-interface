package org.bioimageanalysis.icy.pytorch.tensor;

import org.bioimageanalysis.icy.deeplearning.tensor.RaiArrayUtils;
import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Util;


public class NDarrayBuilder {


    /**
     * Creates a {@link NDArray} from a given {@link Tensor} and an array with its dimensions order.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @param manager
     *        {@link NDManager} needed to create NDArrays
     * @return The NDArray built from the tensor.
     * @throws IllegalArgumentException
     *         If the tensor type is not supported.
     */
    public static NDArray build(Tensor tensor, NDManager manager) throws IllegalArgumentException
    {
        // Create an Icy sequence of the same type of the tensor
    	if (Util.getTypeFromInterval(tensor.getData()) instanceof ByteType) {
            return buildFromTensorByte( tensor.getData(), manager);
    	} else if (Util.getTypeFromInterval(tensor.getData()) instanceof IntType) {
            return buildFromTensorInt( tensor.getData(), manager);
    	} else if (Util.getTypeFromInterval(tensor.getData()) instanceof FloatType) {
            return buildFromTensorFloat( tensor.getData(), manager);
    	} else if (Util.getTypeFromInterval(tensor.getData()) instanceof DoubleType) {
            return buildFromTensorDouble( tensor.getData(), manager);
    	} else {
            throw new IllegalArgumentException("Unsupported tensor type: " + tensor.getDataType());
    	}
    }
    /**
     * Creates a {@link NDArray} from a given {@link RandomAccessibleInterval} and an array with its dimensions order.
     * 
     * @param tensor
     *        The INDArray containing the wanted data.
     * @param manager
     *        {@link NDManager} needed to create NDArrays
     * @return The NDArray built from the tensor.
     * @throws IllegalArgumentException
     *         If the tensor type is not supported.
     */
    public static <T extends Type<T>> NDArray build(RandomAccessibleInterval<T> tensor, NDManager manager) throws IllegalArgumentException
    {
    	if (Util.getTypeFromInterval(tensor) instanceof ByteType) {
            return buildFromTensorByte( (RandomAccessibleInterval<ByteType>) tensor, manager);
    	} else if (Util.getTypeFromInterval(tensor) instanceof IntType) {
            return buildFromTensorInt( (RandomAccessibleInterval<IntType>) tensor, manager);
    	} else if (Util.getTypeFromInterval(tensor) instanceof ByteType) {
            return buildFromTensorFloat( (RandomAccessibleInterval<FloatType>) tensor, manager);
    	} else if (Util.getTypeFromInterval(tensor) instanceof DoubleType) {
            return buildFromTensorDouble( (RandomAccessibleInterval<DoubleType>) tensor, manager);
    	} else {
            throw new IllegalArgumentException("Unsupported tensor type: " + Util.getTypeFromInterval(tensor).getClass().toString());
    	}
    }

    /**
     * Builds a {@link NDArray} from a unsigned byte-typed {@link RandomAccessibleInterval}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @param manager
     *        {@link NDManager} needed to create NDArrays
     * @return The NDArray built from the tensor of type {@link DataType#UBYTE}.
     */
    private static <T extends Type<T>> NDArray buildFromTensorByte(RandomAccessibleInterval<ByteType> tensor, NDManager manager)
    {
    	long[] tensorShape = tensor.dimensionsAsLongArray();
	 	NDArray ndarray = manager.create(RaiArrayUtils.byteArray(tensor), new Shape(tensorShape));
		
		return ndarray;
	}

    /**
     * Builds a {@link NDArray} from a unsigned integer-typed {@link RandomAccessibleInterval}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @param manager
     *        {@link NDManager} needed to create NDArrays
     * @return The NDArray built from the tensor of type {@link DataType#INT}.
     */
    private static <T extends Type<T>> NDArray buildFromTensorInt(RandomAccessibleInterval<IntType> tensor, NDManager manager)
    {
    	long[] tensorShape = tensor.dimensionsAsLongArray();
	 	NDArray ndarray = manager.create(RaiArrayUtils.intArray(tensor), new Shape(tensorShape));
	 	return ndarray;
    }

    /**
     * Builds a {@link NDArray} from a unsigned float-typed {@link RandomAccessibleInterval}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @param manager
     *        {@link NDManager} needed to create NDArrays
     * @return The NDArray built from the tensor of type {@link DataType#FLOAT}.
     */
    private static <T extends Type<T>> NDArray buildFromTensorFloat(RandomAccessibleInterval<FloatType> tensor, NDManager manager)
    {
    	long[] tensorShape = tensor.dimensionsAsLongArray();
	 	NDArray ndarray = manager.create(RaiArrayUtils.floatArray(tensor), new Shape(tensorShape));
	 	return ndarray;
    }

    /**
     * Builds a {@link NDArray} from a unsigned double-typed {@link RandomAccessibleInterval}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @param manager
     *        {@link NDManager} needed to create NDArrays
     * @return The NDArray built from the tensor of type {@link DataType#DOUBLE}.
     */
    private static <T extends Type<T>> NDArray buildFromTensorDouble(RandomAccessibleInterval<DoubleType> tensor, NDManager manager)
    {
    	long[] tensorShape = tensor.dimensionsAsLongArray();
	 	NDArray ndarray = manager.create(RaiArrayUtils.doubleArray(tensor), new Shape(tensorShape));
	 	return ndarray;
    }
}
