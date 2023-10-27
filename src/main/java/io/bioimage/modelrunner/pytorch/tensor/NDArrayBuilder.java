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
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.blocks.PrimitiveBlocks;
import net.imglib2.img.Img;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Util;

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
	public static NDArray build(Tensor tensor, NDManager manager)
		throws IllegalArgumentException
	{
		// Create an Icy sequence of the same type of the tensor
		if (Util.getTypeFromInterval(tensor.getData()) instanceof ByteType) {
			return buildFromTensorByte(tensor.getData(), manager);
		}
		else if (Util.getTypeFromInterval(tensor.getData()) instanceof IntType) {
			return buildFromTensorInt(tensor.getData(), manager);
		}
		else if (Util.getTypeFromInterval(tensor.getData()) instanceof FloatType) {
			return buildFromTensorFloat(tensor.getData(), manager);
		}
		else if (Util.getTypeFromInterval(tensor.getData()) instanceof DoubleType) {
			return buildFromTensorDouble(tensor.getData(), manager);
		}
		else {
			throw new IllegalArgumentException("Unsupported tensor type: " + tensor
				.getDataType());
		}
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
	public static <T extends Type<T>> NDArray build(
		RandomAccessibleInterval<T> tensor, NDManager manager)
		throws IllegalArgumentException
	{
		if (Util.getTypeFromInterval(tensor) instanceof ByteType) {
			return buildFromTensorByte((RandomAccessibleInterval<ByteType>) tensor,
				manager);
		}
		else if (Util.getTypeFromInterval(tensor) instanceof IntType) {
			return buildFromTensorInt((RandomAccessibleInterval<IntType>) tensor,
				manager);
		}
		else if (Util.getTypeFromInterval(tensor) instanceof FloatType) {
			return buildFromTensorFloat((RandomAccessibleInterval<FloatType>) tensor,
				manager);
		}
		else if (Util.getTypeFromInterval(tensor) instanceof DoubleType) {
			return buildFromTensorDouble(
				(RandomAccessibleInterval<DoubleType>) tensor, manager);
		}
		else {
			throw new IllegalArgumentException("Unsupported tensor type: " + Util
				.getTypeFromInterval(tensor).getClass().toString());
		}
	}

	/**
	 * Builds a {@link NDArray} from a signed byte-typed
	 * {@link RandomAccessibleInterval}.
	 * 
	 * @param tensor 
	 * 	the {@link RandomAccessibleInterval} that will be copied into an {@link NDArray}
	 * @param manager 
	 * 	{@link NDManager} needed to create a {@link NDArray}
	 * @return The {@link NDArray} built from the tensor of type {@link ByteType}.
	 */
	private static NDArray buildFromTensorByte(
		RandomAccessibleInterval<ByteType> tensor, NDManager manager)
	{
		long[] ogShape = tensor.dimensionsAsLongArray();
		tensor = Utils.transpose(tensor);
		PrimitiveBlocks< ByteType > blocks = PrimitiveBlocks.of( tensor );
		long[] tensorShape = tensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final byte[] flatArr = new byte[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];
		blocks.copy( tensor.minAsLongArray(), flatArr, sArr );
		NDArray ndarray = manager.create(flatArr, new Shape(ogShape));
		return ndarray;
	}

	/**
	 * Builds a {@link NDArray} from a signed integer-typed
	 * {@link RandomAccessibleInterval}.
	 * 
	 * @param tensor 
	 * 	the {@link RandomAccessibleInterval} that will be copied into an {@link NDArray}
	 * @param manager 
	 * 	{@link NDManager} needed to create a {@link NDArray}
	 * @return The {@link NDArray} built from the tensor of type {@link IntType}.
	 */
	private static NDArray buildFromTensorInt(
		RandomAccessibleInterval<IntType> tensor, NDManager manager)
	{
		long[] ogShape = tensor.dimensionsAsLongArray();
		tensor = Utils.transpose(tensor);
		PrimitiveBlocks< IntType > blocks = PrimitiveBlocks.of( tensor );
		long[] tensorShape = tensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final int[] flatArr = new int[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];
		blocks.copy( tensor.minAsLongArray(), flatArr, sArr );
		NDArray ndarray = manager.create(flatArr, new Shape(ogShape));
		return ndarray;
	}

	/**
	 * Builds a {@link NDArray} from a signed float-typed
	 * {@link RandomAccessibleInterval}.
	 * 
	 * @param tensor 
	 * 	the {@link RandomAccessibleInterval} that will be copied into an {@link NDArray}
	 * @param manager 
	 * 	{@link NDManager} needed to create a {@link NDArray}
	 * @return The {@link NDArray} built from the tensor of type {@link FloatType}.
	 */
	private static NDArray buildFromTensorFloat(
		RandomAccessibleInterval<FloatType> tensor, NDManager manager)
	{
		long[] ogShape = tensor.dimensionsAsLongArray();
		tensor = Utils.transpose(tensor);
		PrimitiveBlocks< FloatType > blocks = PrimitiveBlocks.of( tensor );
		long[] tensorShape = tensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final float[] flatArr = new float[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];
		blocks.copy( tensor.minAsLongArray(), flatArr, sArr );
		NDArray ndarray = manager.create(flatArr, new Shape(ogShape));
		return ndarray;
	}

	/**
	 * Builds a {@link NDArray} from a signed double-typed
	 * {@link RandomAccessibleInterval}.
	 * 
	 * @param tensor 
	 * 	the {@link RandomAccessibleInterval} that will be copied into an {@link NDArray}
	 * @param manager 
	 * 	{@link NDManager} needed to create a {@link NDArray}
	 * @return The {@link NDArray} built from the tensor of type {@link DoubleType}.
	 */
	private static NDArray buildFromTensorDouble(
		RandomAccessibleInterval<DoubleType> tensor, NDManager manager)
	{
		long[] ogShape = tensor.dimensionsAsLongArray();
		tensor = Utils.transpose(tensor);
		PrimitiveBlocks< DoubleType > blocks = PrimitiveBlocks.of( tensor );
		long[] tensorShape = tensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final double[] flatArr = new double[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];
		blocks.copy( tensor.minAsLongArray(), flatArr, sArr );
		NDArray ndarray = manager.create(flatArr, new Shape(ogShape));
		return ndarray;
	}
}
