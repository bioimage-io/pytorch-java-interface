/*-
 * #%L
 * This project complements the DL-model runner acting as the engine that works loading models 
 * 	and making inference with Java 0.3.0 and newer API for Tensorflow 2.
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

package io.bioimage.modelrunner.pytorch.shm;

import io.bioimage.modelrunner.tensor.shm.SharedMemoryArray;
import io.bioimage.modelrunner.utils.CommonUtils;
import net.imglib2.util.Cast;

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.Arrays;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;

/**
 * Utility class to build Pytorch tensors from shm segments using {@link SharedMemoryArray}
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public final class TensorBuilder {

	/**
	 * Utility class.
	 */
	private TensorBuilder() {}

	/**
	 * Creates {@link NDArray} instance from a {@link SharedMemoryArray}
	 * 
	 * @param array
	 * 	the {@link SharedMemoryArray} that is going to be converted into
	 *  a {@link NDArray} tensor
	 * @param manager
	 * 	DJL manager that controls the creation and destruction of {@link NDArrays}
	 * @return the Pytorch {@link NDArray} as the one stored in the shared memory segment
	 * @throws IllegalArgumentException if the type of the {@link SharedMemoryArray}
	 *  is not supported
	 */
	public static NDArray build(SharedMemoryArray array, NDManager manager) throws IllegalArgumentException
	{
		// Create an Icy sequence of the same type of the tensor
		if (array.getOriginalDataType().equals("uint8")) {
			return buildUByte(Cast.unchecked(array), manager);
		}
		else if (array.getOriginalDataType().equals("int32")) {
			return buildInt(Cast.unchecked(array), manager);
		}
		else if (array.getOriginalDataType().equals("float32")) {
			return buildFloat(Cast.unchecked(array), manager);
		}
		else if (array.getOriginalDataType().equals("float64")) {
			return buildDouble(Cast.unchecked(array), manager);
		}
		else if (array.getOriginalDataType().equals("int64")) {
			return buildLong(Cast.unchecked(array), manager);
		}
		else {
			throw new IllegalArgumentException("Unsupported tensor type: " + array.getOriginalDataType());
		}
	}

	private static NDArray buildUByte(SharedMemoryArray tensor, NDManager manager)
		throws IllegalArgumentException
	{
		long[] ogShape = tensor.getOriginalShape();
		if (CommonUtils.int32Overflows(ogShape, 1))
			throw new IllegalArgumentException("Provided tensor with shape " + Arrays.toString(ogShape) 
								+ " is too big. Max number of elements per ubyte tensor supported: " + Integer.MAX_VALUE);
		if (!tensor.isNumpyFormat())
			throw new IllegalArgumentException("Shared memory arrays must be saved in numpy format.");
		ByteBuffer buff = tensor.getDataBufferNoHeader();
		NDArray ndarray = manager.create(buff.array(), new Shape(ogShape));
		return ndarray;
	}

	private static NDArray buildInt(SharedMemoryArray tensor, NDManager manager)
		throws IllegalArgumentException
	{
		long[] ogShape = tensor.getOriginalShape();
		if (CommonUtils.int32Overflows(ogShape, 1))
			throw new IllegalArgumentException("Provided tensor with shape " + Arrays.toString(ogShape) 
								+ " is too big. Max number of elements per ubyte tensor supported: " + Integer.MAX_VALUE);
		if (!tensor.isNumpyFormat())
			throw new IllegalArgumentException("Shared memory arrays must be saved in numpy format.");
		ByteBuffer buff = tensor.getDataBufferNoHeader();
		IntBuffer intBuff = buff.asIntBuffer();
		int[] intArray = new int[intBuff.capacity()];
		intBuff.get(intArray);
		NDArray ndarray = manager.create(intBuff.array(), new Shape(ogShape));
		return ndarray;
	}

	private static NDArray buildLong(SharedMemoryArray tensor, NDManager manager)
		throws IllegalArgumentException
	{
		long[] ogShape = tensor.getOriginalShape();
		if (CommonUtils.int32Overflows(ogShape, 1))
			throw new IllegalArgumentException("Provided tensor with shape " + Arrays.toString(ogShape) 
								+ " is too big. Max number of elements per ubyte tensor supported: " + Integer.MAX_VALUE);
		if (!tensor.isNumpyFormat())
			throw new IllegalArgumentException("Shared memory arrays must be saved in numpy format.");
		ByteBuffer buff = tensor.getDataBufferNoHeader();
		LongBuffer longBuff = buff.asLongBuffer();
		long[] longArray = new long[longBuff.capacity()];
		longBuff.get(longArray);
		NDArray ndarray = manager.create(longBuff.array(), new Shape(ogShape));
		return ndarray;
	}

	private static NDArray buildFloat(SharedMemoryArray tensor, NDManager manager)
		throws IllegalArgumentException
	{
		long[] ogShape = tensor.getOriginalShape();
		if (CommonUtils.int32Overflows(ogShape, 1))
			throw new IllegalArgumentException("Provided tensor with shape " + Arrays.toString(ogShape) 
								+ " is too big. Max number of elements per ubyte tensor supported: " + Integer.MAX_VALUE);
		if (!tensor.isNumpyFormat())
			throw new IllegalArgumentException("Shared memory arrays must be saved in numpy format.");
		ByteBuffer buff = tensor.getDataBufferNoHeader();
		FloatBuffer floatBuff = buff.asFloatBuffer();
		float[] floatArray = new float[floatBuff.capacity()];
		floatBuff.get(floatArray);
		NDArray ndarray = manager.create(floatBuff.array(), new Shape(ogShape));
		return ndarray;
	}

	private static NDArray buildDouble(SharedMemoryArray tensor, NDManager manager)
		throws IllegalArgumentException
	{
		long[] ogShape = tensor.getOriginalShape();
		if (CommonUtils.int32Overflows(ogShape, 1))
			throw new IllegalArgumentException("Provided tensor with shape " + Arrays.toString(ogShape) 
								+ " is too big. Max number of elements per ubyte tensor supported: " + Integer.MAX_VALUE);
		if (!tensor.isNumpyFormat())
			throw new IllegalArgumentException("Shared memory arrays must be saved in numpy format.");
		ByteBuffer buff = tensor.getDataBufferNoHeader();
		DoubleBuffer doubleBuff = buff.asDoubleBuffer();
		double[] doubleArray = new double[doubleBuff.capacity()];
		doubleBuff.get(doubleArray);
		NDArray ndarray = manager.create(doubleBuff.array(), new Shape(ogShape));
		return ndarray;
	}
}
