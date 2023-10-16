/*-
 * #%L
 * This project complements the DL-model runner acting as the engine that works loading models 
 * 	and making inference with Java API for Pytorch.
 * %%
 * Copyright (C) 2022 - 2023 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *      http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * #L%
 */

package io.bioimage.modelrunner.pytorch.tensor;


import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.transform.integer.MixedTransform;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.view.MixedTransformView;
import net.imglib2.view.Views;
import ai.djl.ndarray.NDArray;

/**
* A {@link RandomAccessibleInterval} builder for Pytorch {@link ai.djl.ndarray.NDArray} objects.
* Build ImgLib2 objects (backend of {@link io.bioimage.modelrunner.tensor.Tensor})
* from Pytorch {@link ai.djl.ndarray.NDArray}
* 
* @author Carlos Garcia Lopez de Haro
*/
public class ImgLib2Builder {

	/**
	 * Creates a {@link Img} from a given {@link ai.djl.ndarray.NDArray} 
	 *  
	 * @param <T>
	 * 	the ImgLib2 data type that the {@link Img} can have
	 * @param tensor
	 * 	the {@link ai.djl.ndarray.NDArray} that wants to be converted
	 * @return the {@link Img} that resulted from the {@link ai.djl.ndarray.NDArray} 
	 * @throws IllegalArgumentException if the dataype of the {@link ai.djl.ndarray.NDArray} 
	 * is not supported
	 */
	public static <T extends Type<T>> RandomAccessibleInterval<T> build(NDArray tensor)
		throws IllegalArgumentException
	{
		// Create an ImgLib2 Img of the same type as the NDArray
		switch (tensor.getDataType()) {
			case UINT8:
				return (RandomAccessibleInterval<T>) buildFromTensorUByte(tensor);
			case INT8:
				return (RandomAccessibleInterval<T>) buildFromTensorByte(tensor);
			case INT32:
				return (RandomAccessibleInterval<T>) buildFromTensorInt(tensor);
			case FLOAT32:
				return (RandomAccessibleInterval<T>) buildFromTensorFloat(tensor);
			case FLOAT64:
				return (RandomAccessibleInterval<T>) buildFromTensorDouble(tensor);
			default:
				throw new IllegalArgumentException("Unsupported tensor type: " + tensor
					.getDataType());
		}
	}

	/**
	 * Builds a {@link Img} from a unsigned byte-typed {@link NDArray}.
	 * 
	 * @param tensor 
	 * 	The {@link NDArray} data is read from.
	 * @return The {@link Img} built from the tensor of type {@link UnsignedByteType}.
	 */
	private static RandomAccessibleInterval<UnsignedByteType> buildFromTensorUByte(NDArray tensor) {
		long[] arrayShape = tensor.getShape().getShape();
		long[] tensorShape = new long[arrayShape.length];
		for (int i = 0; i < arrayShape.length; i ++) tensorShape[i] = arrayShape[arrayShape.length - 1 - i];
		byte[] flatArr = tensor.toByteArray();
		RandomAccessibleInterval<UnsignedByteType> rai = ArrayImgs.unsignedBytes(flatArr, tensorShape);
		MixedTransform t = new MixedTransform( tensorShape.length, tensorShape.length );
		int[] transposeAxesOrderChange = new int[tensorShape.length];
		for (int i = 0; i < tensorShape.length; i ++) transposeAxesOrderChange[i] = tensorShape.length - 1 - i;
		t.setComponentMapping(transposeAxesOrderChange);
		long[] minMax = new long[tensorShape.length];
		for (int i = 0; i < tensorShape.length; i ++) minMax[i * 2 + 1] = tensorShape[i];
		return Views.interval(new MixedTransformView<UnsignedByteType>( rai, t ), 
				Intervals.createMinMax(tensorShape));
	}

	/**
	 * Builds a {@link Img} from a signed byte-typed {@link NDArray}.
	 * 
	 * @param tensor 
	 * 	The {@link NDArray} data is read from.
	 * @return The {@link Img} built from the tensor of type {@link ByteType}.
	 */
	private static RandomAccessibleInterval<ByteType> buildFromTensorByte(NDArray tensor) {
		long[] arrayShape = tensor.getShape().getShape();
		long[] tensorShape = new long[arrayShape.length];
		for (int i = 0; i < arrayShape.length; i ++) tensorShape[i] = arrayShape[arrayShape.length - 1 - i];
		byte[] flatArr = tensor.toByteArray();
		RandomAccessibleInterval<ByteType> rai = ArrayImgs.bytes(flatArr, tensorShape);
		MixedTransform t = new MixedTransform( tensorShape.length, tensorShape.length );
		int[] transposeAxesOrderChange = new int[tensorShape.length];
		for (int i = 0; i < tensorShape.length; i ++) transposeAxesOrderChange[i] = tensorShape.length - 1 - i;
		t.setComponentMapping(transposeAxesOrderChange);
		long[] minMax = new long[tensorShape.length];
		for (int i = 0; i < tensorShape.length; i ++) minMax[i * 2 + 1] = tensorShape[i];
		return Views.interval(new MixedTransformView<ByteType>( rai, t ), 
				Intervals.createMinMax(tensorShape));
	}

	/**
	 * Builds a {@link RandomAccessibleInterval} from a signed integer-typed {@link NDArray}.
	 * 
	 * @param tensor 
	 * 	The {@link NDArray} data is read from.
	 * @return The {@link RandomAccessibleInterval} built from the tensor of type {@link IntType}.
	 */
	private static RandomAccessibleInterval<IntType> buildFromTensorInt(NDArray tensor) {
		long[] arrayShape = tensor.getShape().getShape();
		long[] tensorShape = new long[arrayShape.length];
		for (int i = 0; i < arrayShape.length; i ++) tensorShape[i] = arrayShape[arrayShape.length - 1 - i];
		int[] flatArr = tensor.toIntArray();
		RandomAccessibleInterval<IntType> rai = ArrayImgs.ints(flatArr, tensorShape);
		MixedTransform t = new MixedTransform( tensorShape.length, tensorShape.length );
		int[] transposeAxesOrderChange = new int[tensorShape.length];
		for (int i = 0; i < tensorShape.length; i ++) transposeAxesOrderChange[i] = tensorShape.length - 1 - i;
		t.setComponentMapping(transposeAxesOrderChange);
		long[] minMax = new long[tensorShape.length];
		for (int i = 0; i < tensorShape.length; i ++) minMax[i * 2 + 1] = tensorShape[i];
		return Views.interval(new MixedTransformView<IntType>( rai, t ), 
				Intervals.createMinMax(tensorShape));
	}

	/**
	 * Builds a {@link RandomAccessibleInterval} from a signed float-typed {@link NDArray}.
	 * 
	 * @param tensor 
	 * 	The {@link NDArray} data is read from.
	 * @return The {@link RandomAccessibleInterval} built from the tensor of type {@link FloatType}.
	 */
	private static RandomAccessibleInterval<FloatType> buildFromTensorFloat(NDArray tensor) {
		long[] arrayShape = tensor.getShape().getShape();
		long[] tensorShape = new long[arrayShape.length];
		for (int i = 0; i < arrayShape.length; i ++) tensorShape[i] = arrayShape[arrayShape.length - 1 - i];
		float[] flatArr = tensor.toFloatArray();
		RandomAccessibleInterval<FloatType> rai = ArrayImgs.floats(flatArr, tensorShape);
		MixedTransform t = new MixedTransform( tensorShape.length, tensorShape.length );
		int[] transposeAxesOrderChange = new int[tensorShape.length];
		for (int i = 0; i < tensorShape.length; i ++) transposeAxesOrderChange[i] = tensorShape.length - 1 - i;
		t.setComponentMapping(transposeAxesOrderChange);
		long[] minMax = new long[tensorShape.length];
		for (int i = 0; i < tensorShape.length; i ++) minMax[i * 2 + 1] = tensorShape[i];
		return Views.interval(new MixedTransformView<FloatType>( rai, t ), 
				Intervals.createMinMax(tensorShape));
	}

	/**
	 * Builds a {@link RandomAccessibleInterval} from a signed double-typed {@link NDArray}.
	 * 
	 * @param tensor 
	 * 	The {@link NDArray} data is read from.
	 * @return The {@link RandomAccessibleInterval} built from the tensor of type {@link DoubleType}.
	 */
	private static RandomAccessibleInterval<DoubleType> buildFromTensorDouble(NDArray tensor) {
		long[] arrayShape = tensor.getShape().getShape();
		long[] tensorShape = new long[arrayShape.length];
		for (int i = 0; i < arrayShape.length; i ++) tensorShape[i] = arrayShape[arrayShape.length - 1 - i];
		double[] flatArr = tensor.toDoubleArray();
		RandomAccessibleInterval<DoubleType> rai = ArrayImgs.doubles(flatArr, tensorShape);
		MixedTransform t = new MixedTransform( tensorShape.length, tensorShape.length );
		int[] transposeAxesOrderChange = new int[tensorShape.length];
		for (int i = 0; i < tensorShape.length; i ++) transposeAxesOrderChange[i] = tensorShape.length - 1 - i;
		t.setComponentMapping(transposeAxesOrderChange);
		long[] minMax = new long[tensorShape.length];
		for (int i = 0; i < tensorShape.length; i ++) minMax[i * 2 + 1] = tensorShape[i];
		return Views.interval(new MixedTransformView<DoubleType>( rai, t ), 
				Intervals.createMinMax(tensorShape));
	}
}
