package com.icaroerasmo.detectors.shared.engine;

/**
 * Flat float tensor data plus its shape (row-major layout).
 *
 * <p>Both engines (OpenCV DNN and ONNX Runtime) normalize their native output
 * representations into this record so detection post-processing never touches
 * engine-specific types.
 */
public record TensorData(float[] data, long[] shape) {

    public int rank() {
        return shape.length;
    }

    public int size(int dim) {
        return (int) shape[dim];
    }

    public long total() {
        long total = 1;
        for (long s : shape) {
            total *= s;
        }
        return total;
    }
}