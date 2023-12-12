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

package io.bioimage.modelrunner.pytorch.tensor.shm;

import io.bioimage.modelrunner.numpy.DecodeNumpy;
import io.bioimage.modelrunner.pytorch.tensor.ImgLib2Builder;
import io.bioimage.modelrunner.tensor.shm.SharedMemoryArray;
import net.imglib2.util.Cast;

import java.io.IOException;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.util.Map;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;

/**
 * A Pytorch {@link NDArray} builder from shared memory objects
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public class NDArrayShmBuilder {

	/**
	 * Creates a {@link NDArray} from a given the name of a shared memory segment that contains an array
	 * of bytes following the npy format
	 * 
	 * @param memoryName
	 * 	name of the shared memory segment where the data is stored
	 * @param manager
	 *  {@link NDManager} needed to create NDArrays
	 * @return The {@link NDArray} built from the shared memory segment
	 * @throws IllegalArgumentException if the data type is not supported by {@link NDArray}
	 */
	public static NDArray buildFromShma(String memoryName, NDManager manager) throws IllegalArgumentException
	{
		
		Map<String, Object> map = SharedMemoryArray.buildMapFromNumpyLikeSHMA(memoryName);

		String dtype = (String) map.get(DecodeNumpy.DTYPE_KEY);
		ByteBuffer data = (ByteBuffer) map.get(DecodeNumpy.DATA_KEY);
		long[] shape = (long[]) map.get(DecodeNumpy.SHAPE_KEY);
		Buffer buff = null;
		if (dtype.equals("int32")) {
			buff = data.asIntBuffer();
		} else if (dtype.equals("float32")) {
			buff = data.asFloatBuffer();
		} else if (dtype.equals("float16")) {
			buff = data.asShortBuffer();
		} else if (dtype.equals("float64")) {
			buff = data.asDoubleBuffer();
		} else {
			throw new IllegalArgumentException("Unsupported tensor type: " + dtype);
		}
		NDArray ndarray = manager.create(buff, new Shape(shape));
		return ndarray;
	}
	
	public static SharedMemoryArray buildShma(NDArray tensor, String memoryName) throws IOException {
		return SharedMemoryArray.buildNumpyLikeSHMA(memoryName, Cast.unchecked(ImgLib2Builder.build(tensor)));
	}
}
