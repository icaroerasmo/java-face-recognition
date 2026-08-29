package com.icaroerasmo.detectors.shared.engine;

import com.icaroerasmo.properties.AccelerationProperties;
import com.icaroerasmo.utils.OpenCvResourceHelper;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Engine selection tests. These load the real YOLOv8n model and run a validation
 * forward pass, so they are hermetic (CPU-only) and pass on any machine.
 */
class DnnEngineFactoryTest {

    private static final String MODEL = "opencv/yolov8n.onnx";

    private static String modelPath() throws Exception {
        return OpenCvResourceHelper.getResourcePath(MODEL, DnnEngineFactoryTest.class);
    }

    @Test
    void onnxCpuBackendSelectsOnnxRuntimeEngine() throws Exception {
        try (DnnEngine engine = DnnEngineFactory.create(
                modelPath(), AccelerationProperties.Backend.ONNX_CPU, null, true, "test")) {
            assertNotNull(engine);
            assertTrue(engine.describe().contains("ONNX Runtime"), "expected ONNX Runtime, got: " + engine.describe());
        }
    }

    @Test
    void opencvBackendSelectsOpenCvDnnEngine() throws Exception {
        try (DnnEngine engine = DnnEngineFactory.create(
                modelPath(), AccelerationProperties.Backend.OPENCV, AccelerationProperties.Target.CPU, true, "test")) {
            assertNotNull(engine);
            assertTrue(engine.describe().contains("OpenCV DNN"), "expected OpenCV DNN, got: " + engine.describe());
        }
    }

    @Test
    void autoBackendSelectsOpenCvDnnEngine() throws Exception {
        try (DnnEngine engine = DnnEngineFactory.create(
                modelPath(), AccelerationProperties.Backend.AUTO, AccelerationProperties.Target.AUTO, true, "test")) {
            assertNotNull(engine);
            assertTrue(engine.describe().contains("OpenCV DNN"), "expected OpenCV DNN, got: " + engine.describe());
        }
    }

    @Test
    void cudaBackendFallsBackToCpuWhenUnavailable() throws Exception {
        // On machines without a working CUDA DNN backend the engine must still be
        // created (fallback to CPU), never throw and never loop.
        try (DnnEngine engine = DnnEngineFactory.create(
                modelPath(), AccelerationProperties.Backend.CUDA, AccelerationProperties.Target.CUDA, true, "test")) {
            assertNotNull(engine);
            assertTrue(engine.describe().contains("OpenCV DNN"), "expected OpenCV DNN, got: " + engine.describe());
        }
    }
}