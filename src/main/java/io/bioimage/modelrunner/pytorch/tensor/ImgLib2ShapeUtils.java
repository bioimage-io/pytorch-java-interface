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

/**
 * Class that provides methods to manage the shape of DJL NDArrays
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public class ImgLib2ShapeUtils {

	/**
	 * Converts an int array into a long array
	 * 
	 * @param shapeArr 
	 * 	int array 
	 * @return long array copied from the int array
	 */
	public static long[] intArrayToLongArray(int[] shapeArr) {
		long[] dimensionSizes = new long[shapeArr.length];
		for (int i = 0; i < dimensionSizes.length; i++) {
			dimensionSizes[i] = (long) shapeArr[i];
		}
		return dimensionSizes;
	}

}
