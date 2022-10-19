package org.bioimageanalysis.icy.pytorch.tensor;

import org.bioimageanalysis.icy.deeplearning.utils.IndexingUtils;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.Shape;
import net.imglib2.Cursor;
import net.imglib2.img.Img;
import net.imglib2.img.ImgFactory;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.cell.CellImgFactory;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.IntervalView;

public class ImgLib2Builder {


    /**
     * Creates a {@link Img} from a given {@link Tensor} and an array with its dimensions order.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor.
     * @throws IllegalArgumentException
     *         If the tensor type is not supported.
     */
    public static <T extends Type<T>> Img<T> build(NDArray tensor) throws IllegalArgumentException
    {
        // Create an Icy sequence of the same type of the tensor
        switch (tensor.getDataType())
        {
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
                throw new IllegalArgumentException("Unsupported tensor type: " + tensor.getDataType());
        }
    }

    /**
     * Builds a {@link Img} from a unsigned byte-typed {@link NDArray}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor of type {@link DataType#UBYTE}.
     */
    private static Img<ByteType> buildFromTensorByte(NDArray tensor)
    {
		long[] tensorShape = tensor.getShape().getShape();
		return ArrayImgs.bytes(tensor.toByteArray(), tensorShape);
	}

    /**
     * Builds a {@link Img} from a unsigned integer-typed {@link NDArray}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor of type {@link DataType#INT}.
     */
    private static Img<IntType> buildFromTensorInt(NDArray tensor)
    {
		long[] tensorShape = tensor.getShape().getShape();
		return ArrayImgs.ints(tensor.toIntArray(), tensorShape);
    }

    /**
     * Builds a {@link Img} from a unsigned float-typed {@link Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor of type {@link DataType#FLOAT}.
     */
    private static Img<FloatType> buildFromTensorFloat(NDArray tensor)
    {
    	long[] tensorShape = tensor.getShape().getShape();
    	final ImgFactory< FloatType > factory = new CellImgFactory<>( new FloatType(), 5 );
        final Img< FloatType > outputImg = (Img<FloatType>) factory.create(tensorShape);
    	Cursor<FloatType> tensorCursor= outputImg.cursor();
		float[] flatArr = tensor.toFloatArray();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, tensorShape);
        	float val = flatArr[flatPos];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    	/*
		long[] tensorShape = tensor.getShape().getShape();
		return ArrayImgs.floats(tensor.toFloatArray(), tensorShape);
		*/
    }

    /**
     * Builds a {@link Img} from a unsigned double-typed {@link NDArray}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor of type {@link DataType#DOUBLE}.
     */
    private static Img<DoubleType> buildFromTensorDouble(NDArray tensor)
    {
		long[] tensorShape = tensor.getShape().getShape();
		return ArrayImgs.doubles(tensor.toDoubleArray(), tensorShape);
    }
}
