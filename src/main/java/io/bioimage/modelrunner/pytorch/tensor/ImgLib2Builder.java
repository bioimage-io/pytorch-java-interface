/*-
 * #%L
 * This project complements the DL-model runner acting as the engine that works loading models 
 * 	and making inference with Java API for Pytorch.
 * %%
 * Copyright (C) 2022 - 2023 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 * #L%
 */

package io.bioimage.modelrunner.pytorch.tensor;

import io.bioimage.modelrunner.utils.IndexingUtils;

import net.imglib2.Cursor;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;

import ai.djl.ndarray.NDArray;

/**
* A {@link Img} builder for Pytorch {@link ai.djl.ndarray.NDArray} objects.
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
	public static <T extends Type<T>> Img<T> build(NDArray tensor)
		throws IllegalArgumentException
	{
		// Create an ImgLib2 Img of the same type as the NDArray
		switch (tensor.getDataType()) {
			case UINT8:
				return (Img<T>) buildFromTensorUByte(tensor);
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
	 * @param tensor 
	 * 	The {@link NDArray} data is read from.
	 * @return The {@link Img} built from the tensor of type {@link UnsignedByteType}.
	 */
	private static Img<UnsignedByteType> buildFromTensorUByte(NDArray tensor) {
		long[] tensorShape = tensor.getShape().getShape();
		final ArrayImgFactory<UnsignedByteType> factory = new ArrayImgFactory<>(new UnsignedByteType());
		final Img<UnsignedByteType> outputImg = factory.create(tensorShape);
		Cursor<UnsignedByteType> tensorCursor = outputImg.cursor();
		byte[] flatArr = tensor.toByteArray();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
			int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos,
				tensorShape);
			byte val = flatArr[flatPos];
			if (val < 0)
				tensorCursor.get().set(256 + (int) val);
			else
				tensorCursor.get().set(val);
		}
		return outputImg;
	}

	/**
	 * Builds a {@link Img} from a signed byte-typed {@link NDArray}.
	 * 
	 * @param tensor 
	 * 	The {@link NDArray} data is read from.
	 * @return The {@link Img} built from the tensor of type {@link ByteType}.
	 */
	private static Img<ByteType> buildFromTensorByte(NDArray tensor) {
		long[] tensorShape = tensor.getShape().getShape();
		final ArrayImgFactory<ByteType> factory = new ArrayImgFactory<>(new ByteType());
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
	 * Builds a {@link Img} from a signed integer-typed {@link NDArray}.
	 * 
	 * @param tensor 
	 * 	The {@link NDArray} data is read from.
	 * @return The {@link Img} built from the tensor of type {@link IntType}.
	 */
	private static Img<IntType> buildFromTensorInt(NDArray tensor) {
		long[] tensorShape = tensor.getShape().getShape();
		final ArrayImgFactory<IntType> factory = new ArrayImgFactory<>(new IntType());
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
	 * Builds a {@link Img} from a signed float-typed {@link NDArray}.
	 * 
	 * @param tensor 
	 * 	The {@link NDArray} data is read from.
	 * @return The {@link Img} built from the tensor of type {@link FloatType}.
	 */
	private static Img<FloatType> buildFromTensorFloat(NDArray tensor) {
		long[] tensorShape = tensor.getShape().getShape();
		final ArrayImgFactory<FloatType> factory = new ArrayImgFactory<>(new FloatType());
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
	 * Builds a {@link Img} from a signed double-typed {@link NDArray}.
	 * 
	 * @param tensor 
	 * 	The {@link NDArray} data is read from.
	 * @return The {@link Img} built from the tensor of type {@link DoubleType}.
	 */
	private static Img<DoubleType> buildFromTensorDouble(NDArray tensor) {
		long[] tensorShape = tensor.getShape().getShape();
		final ArrayImgFactory<DoubleType> factory = new ArrayImgFactory<>(new DoubleType());
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
