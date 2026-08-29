package com.icaroerasmo.detectors.shared.engine;

import com.icaroerasmo.properties.AccelerationProperties;
import com.icaroerasmo.utils.OpenCvResourceHelper;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.CV_32F;
import static org.bytedeco.opencv.global.opencv_core.CV_32FC3;
import static org.bytedeco.opencv.global.opencv_dnn.blobFromImage;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * ONNX Runtime engine forward-pass tests (CPU EP, hermetic).
 */
class OnnxRuntimeEngineTest {

    private static final String MODEL = "opencv/yolov8n.onnx";

    @Test
    void forwardReturnsExpectedYoloShape() throws Exception {
        String modelPath = OpenCvResourceHelper.getResourcePath(MODEL, OnnxRuntimeEngineTest.class);
        try (DnnEngine engine = new OnnxRuntimeEngine(
                modelPath, AccelerationProperties.Backend.ONNX_CPU, true, "test")) {

            Mat dummy = null, blob = null;
            Size size = null;
            Scalar mean = null;
            try {
                dummy = new Mat(640, 640, CV_32FC3, new Scalar(0, 0, 0, 0));
                size = new Size(640, 640);
                mean = new Scalar(0, 0, 0, 0);
                blob = blobFromImage(dummy, 1.0 / 255.0, size, mean, true, false, CV_32F);

                TensorData output = engine.forward(blob);
                assertNotNull(output);
                assertEquals(3, output.rank());
                assertEquals(1, output.size(0));
                assertEquals(84, output.size(1));
                assertEquals(8400, output.size(2));
                assertEquals(84L * 8400L, output.total());
                assertEquals(output.total(), output.data().length);
            } finally {
                if (blob != null) {
                    blob.deallocate();
                }
                if (dummy != null) {
                    dummy.deallocate();
                }
                if (size != null) {
                    size.deallocate();
                }
                if (mean != null) {
                    mean.deallocate();
                }
            }
        }
    }

    @Test
    void getOutputNamesContainsOutput0() throws Exception {
        String modelPath = OpenCvResourceHelper.getResourcePath(MODEL, OnnxRuntimeEngineTest.class);
        try (DnnEngine engine = new OnnxRuntimeEngine(
                modelPath, AccelerationProperties.Backend.ONNX_CPU, true, "test")) {
            List<String> names = engine.getOutputNames();
            assertNotNull(names);
            assertFalse(names.isEmpty());
            assertTrue(names.stream().anyMatch(n -> n.contains("output")), "expected an output name, got: " + names);
        }
    }
}