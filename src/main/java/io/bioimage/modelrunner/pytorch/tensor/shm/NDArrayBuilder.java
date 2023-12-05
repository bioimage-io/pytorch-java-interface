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
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.tensor.Utils;
import io.bioimage.modelrunner.utils.CommonUtils;
import net.imglib2.Cursor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.blocks.PrimitiveBlocks;
import net.imglib2.img.Img;
import net.imglib2.type.NativeType;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Util;
import net.imglib2.view.Views;

import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.HashMap;
import java.util.regex.Matcher;

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
		if (CommonUtils.int32Overflows(ogShape))
			throw new IllegalArgumentException("Provided tensor with shape " + Arrays.toString(ogShape) 
								+ " is too big. Max number of elements per tensor supported: " + Integer.MAX_VALUE);
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
		if (CommonUtils.int32Overflows(ogShape))
			throw new IllegalArgumentException("Provided tensor with shape " + Arrays.toString(ogShape) 
								+ " is too big. Max number of elements per tensor supported: " + Integer.MAX_VALUE);
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
		if (CommonUtils.int32Overflows(ogShape))
			throw new IllegalArgumentException("Provided tensor with shape " + Arrays.toString(ogShape) 
								+ " is too big. Max number of elements per tensor supported: " + Integer.MAX_VALUE);
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
		if (CommonUtils.int32Overflows(ogShape))
			throw new IllegalArgumentException("Provided tensor with shape " + Arrays.toString(ogShape) 
								+ " is too big. Max number of elements per tensor supported: " + Integer.MAX_VALUE);
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
    
    /**
     * MEthod to decode the bytes corresponding to a numpy array stored in the numpy file
     * @param <T>
     * 	possible data types that the ImgLib2 image can have
     * @param is
     * 	{@link InputStream} that results after reading the numpy file. Contains the byte info of the
     * 	numpy array
     * @return an ImgLib2 image with the same datatype, shape and data that the numpy array
     * @throws IOException if there is any error reading the {@link InputStream}
     */
    public static < T extends RealType< T > & NativeType< T > > 
    				HashMap<String, Object> decodeNumpyFromByteArrayStream(InputStream is) throws IOException {
        DataInputStream dis;
        if (is instanceof DataInputStream) {
            dis = (DataInputStream) is;
        } else {
            dis = new DataInputStream(is);
        }

        byte[] buf = new byte[DecodeNumpy.MAGIC_PREFIX.length];
        dis.readFully(buf);
        if (!Arrays.equals(buf, DecodeNumpy.MAGIC_PREFIX)) {
            throw new IllegalArgumentException("Malformed  or unsopported Numpy array");
        }
        byte major = dis.readByte();
        byte minor = dis.readByte();
        if (major < 1 || major > 3 || minor != 0) {
            throw new IllegalArgumentException("Unknown numpy version: " + major + '.' + minor);
        }
        int len = major == 1 ? 2 : 4;
        dis.readFully(buf, 0, len);
        ByteBuffer bb = ByteBuffer.wrap(buf, 0, len);
        bb.order(ByteOrder.LITTLE_ENDIAN);
        if (major == 1) {
            len = bb.getShort();
        } else {
            len = bb.getInt();
        }
        buf = new byte[len];
        dis.readFully(buf);
        String header = new String(buf, StandardCharsets.UTF_8);
        Matcher m = DecodeNumpy.HEADER_PATTERN.matcher(header);
        if (!m.find()) {
            throw new IllegalArgumentException("Invalid numpy header: " + header);
        }
        String typeStr = m.group(1);
        String fortranOrder = m.group(2).trim();
        String shapeStr = m.group(3);
        long[] shape = new long[0];
        if (!shapeStr.isEmpty()) {
            String[] tokens = shapeStr.split(", ?");
            shape = Arrays.stream(tokens).mapToLong(Long::parseLong).toArray();
        }
        char order = typeStr.charAt(0);
        ByteOrder byteOrder = null;
        if (order == '>') {
        	byteOrder = ByteOrder.BIG_ENDIAN;
        } else if (order == '<') {
        	byteOrder = ByteOrder.LITTLE_ENDIAN;
        } else if (order == '|') {
        	byteOrder = ByteOrder.LITTLE_ENDIAN;
        	new IOException("Numpy .npy file did not specify the byte order of the array."
        			+ " It was automatically opened as little endian but this does not guarantee"
        			+ " the that the file is open correctly. Caution is advised.").printStackTrace();
    	} else {
        	new IllegalArgumentException("Not supported ByteOrder for the provided .npy array.");
        }
        String dtype = DecodeNumpy.getDataType(typeStr.substring(1));
        long numBytes = DecodeNumpy.DATA_TYPES_MAP.get(dtype);
    	long count;
    	if (shape.length == 0)
    		count = 1;
		else
			count = Arrays.stream(shape).reduce(Math::multiplyExact).getAsLong();
        //len = Math.toIntExact(shape.length * numBytes);
        len = Math.toIntExact(count * numBytes);
        ByteBuffer data = ByteBuffer.allocate(len);
        data.order(byteOrder);
        readData(dis, data, len);
        
        HashMap<String, Object> map = new HashMap<String, Object>();
        map.put("shape", shape);
        map.put("byte_order", byteOrder);
        map.put("dtype", dtype);
        map.put("fortran_order", fortranOrder.equals("True"));
        map.put("data", data);

        return map;
    }

    /**
     * Read the data from the input stream into a byte buffer
     * @param dis
     * 	the {@link DataInputStream} from where the data is read from
     * @param data
     * 	{@link ByteBuffer} where the info is copied to
     * @param len
     * 	remaining number of bytes in the {@link DataInputStream}
     * @throws IOException if there is any error reading the {@link DataInputStream}
     */
    private static void readData(DataInputStream dis, ByteBuffer data, int len) throws IOException {
        if (len > 0) {
            byte[] buf = new byte[DecodeNumpy.BUFFER_SIZE];
            while (len > DecodeNumpy.BUFFER_SIZE) {
                dis.readFully(buf);
                data.put(buf);
                len -= DecodeNumpy.BUFFER_SIZE;
            }

            dis.readFully(buf, 0, len);
            data.put(buf, 0, len);
            data.rewind();
        }
    }
}
