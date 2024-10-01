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

import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.tensor.Utils;
import io.bioimage.modelrunner.utils.CommonUtils;
import net.imglib2.Cursor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Cast;
import net.imglib2.util.Util;
import net.imglib2.view.Views;

import java.util.Arrays;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;

/**
 * A Pytorch {@link NDArray} builder for {@link Img} and
 * {@link io.bioimage.modelrunner.tensor.Tensor} objects.
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public class NDArrayBuilder {

	/**
	 * Creates a {@link NDArray} from a given {@link Tensor}. The {@link Tensor}
	 * contains the data and info(dimensions, dataype) necessary to build a {@link NDArray}
	 * 
	 * @param tensor 
	 * 	The {@link Tensor} that will be copied into a {@link NDArray}
	 * @param manager
	 *  {@link NDManager} needed to create NDArrays
	 * @return The {@link NDArray} built from the {@link Tensor}.
	 * @throws IllegalArgumentException If the tensor type is not supported.
	 */
	public static <T extends RealType<T> & NativeType<T>> 
	NDArray build(Tensor<T> tensor, NDManager manager) throws IllegalArgumentException {
		return build(tensor.getData(), manager);
	}

	/**
	 * Creates a {@link NDArray} from a given {@link RandomAccessibleInterval}.
	 * 
	 * @param <T>
	 * 	possible ImgLib2 datatypes of the {@link RandomAccessibleInterval}
	 * @param tensor
	 * 	the {@link RandomAccessibleInterval} that will be copied into an {@link NDArray}
	 * @param manager
	 *  {@link NDManager} needed to create NDArrays
	 * @return The {@link NDArray} built from the {@link RandomAccessibleInterval}.
	 * @throws IllegalArgumentException if the {@link RandomAccessibleInterval} is not supported
	 */
	public static <T extends RealType<T> & NativeType<T>> 
	NDArray build(RandomAccessibleInterval<T> tensor, NDManager manager)
		throws IllegalArgumentException
	{
		if (Util.getTypeFromInterval(tensor) instanceof ByteType) {
			return buildFromTensorByte(Cast.unchecked(tensor), manager);
		}
		else if (Util.getTypeFromInterval(tensor) instanceof IntType) {
			return buildFromTensorInt(Cast.unchecked(tensor), manager);
		}
		else if (Util.getTypeFromInterval(tensor) instanceof FloatType) {
			return buildFromTensorFloat(Cast.unchecked(tensor), manager);
		}
		else if (Util.getTypeFromInterval(tensor) instanceof DoubleType) {
			return buildFromTensorDouble(Cast.unchecked(tensor), manager);
		}
		else {
			throw new IllegalArgumentException("Unsupported tensor type: " + Util
				.getTypeFromInterval(tensor).getClass().toString());
		}
	}

	private static NDArray buildFromTensorByte(
		RandomAccessibleInterval<ByteType> tensor, NDManager manager)
	{
		long[] ogShape = tensor.dimensionsAsLongArray();
		if (CommonUtils.int32Overflows(ogShape, 1))
			throw new IllegalArgumentException("Provided tensor with shape " + Arrays.toString(ogShape) 
								+ " is too big. Max number of elements per byte tensor supported: " + Integer.MAX_VALUE);
		tensor = Utils.transpose(tensor);
		long[] tensorShape = tensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final byte[] flatArr = new byte[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];

		Cursor<ByteType> cursor = Views.flatIterable(tensor).cursor();
		int i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			flatArr[i ++] = cursor.get().getByte();
		}
		NDArray ndarray = manager.create(flatArr, new Shape(ogShape));
		return ndarray;
	}

	private static NDArray buildFromTensorInt(
		RandomAccessibleInterval<IntType> tensor, NDManager manager)
	{
		long[] ogShape = tensor.dimensionsAsLongArray();
		if (CommonUtils.int32Overflows(ogShape, 4))
			throw new IllegalArgumentException("Provided tensor with shape " + Arrays.toString(ogShape) 
								+ " is too big. Max number of elements per int tensor supported: " + Integer.MAX_VALUE);
		tensor = Utils.transpose(tensor);
		long[] tensorShape = tensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final int[] flatArr = new int[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];

		Cursor<IntType> cursor = Views.flatIterable(tensor).cursor();
		int i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			flatArr[i ++] = cursor.get().getInt();
		}
		NDArray ndarray = manager.create(flatArr, new Shape(ogShape));
		return ndarray;
	}

	private static NDArray buildFromTensorFloat(
		RandomAccessibleInterval<FloatType> tensor, NDManager manager)
	{
		long[] ogShape = tensor.dimensionsAsLongArray();
		if (CommonUtils.int32Overflows(ogShape, 4))
			throw new IllegalArgumentException("Provided tensor with shape " + Arrays.toString(ogShape) 
								+ " is too big. Max number of elements per float tensor supported: " + Integer.MAX_VALUE);
		tensor = Utils.transpose(tensor);
		long[] tensorShape = tensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final float[] flatArr = new float[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];

		Cursor<FloatType> cursor = Views.flatIterable(tensor).cursor();
		int i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			flatArr[i ++] = cursor.get().get();
		}
		NDArray ndarray = manager.create(flatArr, new Shape(ogShape));
		return ndarray;
	}

	private static NDArray buildFromTensorDouble(
		RandomAccessibleInterval<DoubleType> tensor, NDManager manager)
	{
		long[] ogShape = tensor.dimensionsAsLongArray();
		if (CommonUtils.int32Overflows(ogShape, 8))
			throw new IllegalArgumentException("Provided tensor with shape " + Arrays.toString(ogShape) 
								+ " is too big. Max number of elements per double tensor supported: " + Integer.MAX_VALUE);
		tensor = Utils.transpose(tensor);
		long[] tensorShape = tensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final double[] flatArr = new double[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];

		Cursor<DoubleType> cursor = Views.flatIterable(tensor).cursor();
		int i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			flatArr[i ++] = cursor.get().get();
		}
		NDArray ndarray = manager.create(flatArr, new Shape(ogShape));
		return ndarray;
	}
}
