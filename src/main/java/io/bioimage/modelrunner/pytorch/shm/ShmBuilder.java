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

import io.bioimage.modelrunner.system.PlatformDetection;
import io.bioimage.modelrunner.tensor.shm.SharedMemoryArray;
import io.bioimage.modelrunner.utils.CommonUtils;

import java.io.IOException;
import java.util.Arrays;

import ai.djl.ndarray.NDArray;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;

/**
 * A utility class that converts {@link NDArray}s into {@link SharedMemoryArray}s for
 * interprocessing communication
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public final class ShmBuilder
{
    /**
     * Utility class.
     */
    private ShmBuilder()
    {
    }

    /**
     * Create a {@link SharedMemoryArray} from a {@link NDArray}
     * @param tensor
     * 	the tensor to be passed into the other process through the shared memory
     * @param memoryName
     * 	the name of the memory region where the tensor is going to be copied
     * @throws IllegalArgumentException if the data type of the tensor is not supported
     * @throws IOException if there is any error creating the shared memory array
     */
	public static void build(NDArray  tensor, String memoryName) throws IllegalArgumentException, IOException
    {
		switch (tensor.getDataType())
        {
            case UINT8:
            	buildFromTensorUByte(tensor, memoryName);
            	break;
            case INT32:
            	buildFromTensorInt(tensor, memoryName);
            	break;
            case FLOAT32:
            	buildFromTensorFloat(tensor, memoryName);
            	break;
            case FLOAT64:
            	buildFromTensorDouble(tensor, memoryName);
            	break;
            case INT64:
            	buildFromTensorLong(tensor, memoryName);
            	break;
            default:
                throw new IllegalArgumentException("Unsupported tensor type: " + tensor.getDataType().asNumpy());
        }
    }

    private static void buildFromTensorUByte(NDArray tensor, String memoryName) throws IOException
    {
    	long[] arrayShape = tensor.getShape().getShape();
		if (CommonUtils.int32Overflows(arrayShape, 1))
			throw new IllegalArgumentException("Model output tensor with shape " + Arrays.toString(arrayShape) 
					+ " is too big. Max number of elements per ubyte output tensor supported: " + Integer.MAX_VALUE / 1);
        SharedMemoryArray shma = SharedMemoryArray.readOrCreate(memoryName, arrayShape, new UnsignedByteType(), true, true);
        shma.getDataBufferNoHeader().put(tensor.toByteArray());
        if (PlatformDetection.isWindows()) shma.close();
    }

    private static void buildFromTensorInt(NDArray tensor, String memoryName) throws IOException
    {
    	long[] arrayShape = tensor.getShape().getShape();
		if (CommonUtils.int32Overflows(arrayShape, 4))
			throw new IllegalArgumentException("Model output tensor with shape " + Arrays.toString(arrayShape) 
					+ " is too big. Max number of elements per int output tensor supported: " + Integer.MAX_VALUE / 4);

        SharedMemoryArray shma = SharedMemoryArray.readOrCreate(memoryName, arrayShape, new IntType(), true, true);
        shma.getDataBufferNoHeader().put(tensor.toByteArray());
        if (PlatformDetection.isWindows()) shma.close();
    }

    private static void buildFromTensorFloat(NDArray tensor, String memoryName) throws IOException
    {
    	long[] arrayShape = tensor.getShape().getShape();
		if (CommonUtils.int32Overflows(arrayShape, 4))
			throw new IllegalArgumentException("Model output tensor with shape " + Arrays.toString(arrayShape) 
					+ " is too big. Max number of elements per float output tensor supported: " + Integer.MAX_VALUE / 4);

        SharedMemoryArray shma = SharedMemoryArray.readOrCreate(memoryName, arrayShape, new FloatType(), true, true);
        shma.getDataBufferNoHeader().put(tensor.toByteArray());
        if (PlatformDetection.isWindows()) shma.close();
    }

    private static void buildFromTensorDouble(NDArray tensor, String memoryName) throws IOException
    {
    	long[] arrayShape = tensor.getShape().getShape();
		if (CommonUtils.int32Overflows(arrayShape, 8))
			throw new IllegalArgumentException("Model output tensor with shape " + Arrays.toString(arrayShape) 
					+ " is too big. Max number of elements per double output tensor supported: " + Integer.MAX_VALUE / 8);

        SharedMemoryArray shma = SharedMemoryArray.readOrCreate(memoryName, arrayShape, new DoubleType(), true, true);
        shma.getDataBufferNoHeader().put(tensor.toByteArray());
        if (PlatformDetection.isWindows()) shma.close();
    }

    private static void buildFromTensorLong(NDArray tensor, String memoryName) throws IOException
    {
    	long[] arrayShape = tensor.getShape().getShape();
		if (CommonUtils.int32Overflows(arrayShape, 8))
			throw new IllegalArgumentException("Model output tensor with shape " + Arrays.toString(arrayShape) 
					+ " is too big. Max number of elements per long output tensor supported: " + Integer.MAX_VALUE / 8);
		

        SharedMemoryArray shma = SharedMemoryArray.readOrCreate(memoryName, arrayShape, new LongType(), true, true);
        shma.getDataBufferNoHeader().put(tensor.toByteArray());
        if (PlatformDetection.isWindows()) shma.close();
    }
}
