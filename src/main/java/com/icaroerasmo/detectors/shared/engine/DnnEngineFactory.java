package com.icaroerasmo.detectors.shared.engine;

import com.icaroerasmo.properties.AccelerationProperties;

/**
 * Creates the appropriate {@link DnnEngine} for the configured backend.
 *
 * <p>Watchdog: selection is deterministic and never retries. Each engine performs at
 * most one fallback (accelerated -&gt; CPU) at construction time; there is no retry loop.
 */
public final class DnnEngineFactory {

    private DnnEngineFactory() {
    }

    /**
     * @param modelPath     path to the ONNX model file
     * @param backend       configured backend ({@code null} or {@code AUTO} = auto-detect)
     * @param target        OpenCV DNN target (only used for OpenCV-based backends)
     * @param fallbackToCpu whether to fall back to CPU when the requested backend fails
     * @param modelName     human-readable model name for logging
     */
    public static DnnEngine create(
            String modelPath,
            AccelerationProperties.Backend backend,
            AccelerationProperties.Target target,
            boolean fallbackToCpu,
            String modelName
    ) {
        AccelerationProperties.Backend effective = (backend == null) ? AccelerationProperties.Backend.AUTO : backend;
        if (isOnnxBackend(effective)) {
            return new OnnxRuntimeEngine(modelPath, effective, fallbackToCpu, modelName);
        }
        return new OpenCvDnnEngine(modelPath, effective, target, fallbackToCpu, modelName);
    }

    private static boolean isOnnxBackend(AccelerationProperties.Backend backend) {
        return backend == AccelerationProperties.Backend.ONNX_CPU
                || backend == AccelerationProperties.Backend.ONNX_CUDA
                || backend == AccelerationProperties.Backend.ONNX_ROCM;
    }
}