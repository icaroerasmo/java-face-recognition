package com.icaroerasmo.detectors.shared.engine;

import com.icaroerasmo.properties.AccelerationProperties;
import lombok.extern.log4j.Log4j2;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.opencv.opencv_core.IntIntPairVector;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.MatVector;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_core.StringVector;
import org.bytedeco.opencv.opencv_dnn.Net;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.bytedeco.opencv.global.opencv_core.CV_32F;
import static org.bytedeco.opencv.global.opencv_core.CV_32FC3;
import static org.bytedeco.opencv.global.opencv_dnn.DNN_BACKEND_CUDA;
import static org.bytedeco.opencv.global.opencv_dnn.DNN_BACKEND_OPENCV;
import static org.bytedeco.opencv.global.opencv_dnn.DNN_BACKEND_VKCOM;
import static org.bytedeco.opencv.global.opencv_dnn.DNN_TARGET_CPU;
import static org.bytedeco.opencv.global.opencv_dnn.DNN_TARGET_CUDA;
import static org.bytedeco.opencv.global.opencv_dnn.DNN_TARGET_CUDA_FP16;
import static org.bytedeco.opencv.global.opencv_dnn.DNN_TARGET_OPENCL;
import static org.bytedeco.opencv.global.opencv_dnn.DNN_TARGET_OPENCL_FP16;
import static org.bytedeco.opencv.global.opencv_dnn.DNN_TARGET_VULKAN;
import static org.bytedeco.opencv.global.opencv_dnn.blobFromImage;
import static org.bytedeco.opencv.global.opencv_dnn.getAvailableBackends;
import static org.bytedeco.opencv.global.opencv_dnn.readNetFromONNX;

/**
 * OpenCV DNN engine. Supports CPU, CUDA, OpenCL and Vulkan targets.
 *
 * <p>Watchdog: the requested backend is validated at construction time with a single
 * dummy forward pass. If it fails and {@code fallbackToCpu} is enabled, the network is
 * reloaded on CPU exactly once - never more. There is no retry loop.
 */
@Log4j2
public class OpenCvDnnEngine implements DnnEngine {

    private static final int TEST_INPUT_SIZE = 640;

    private final Net net;
    private final String description;

    public OpenCvDnnEngine(
            String modelPath,
            AccelerationProperties.Backend backend,
            AccelerationProperties.Target target,
            boolean fallbackToCpu,
            String modelName
    ) {
        BackendTarget resolved = resolveBackendTarget(backend, target);
        Net createdNet;
        String createdDescription;
        Net candidate = loadNet(modelPath);
        try {
            candidate.setPreferableBackend(resolved.backend());
            candidate.setPreferableTarget(resolved.target());
            validateForward(candidate, modelName, resolved);
            createdNet = candidate;
            createdDescription = "OpenCV DNN (backend=" + resolved.backendName() + ", target=" + resolved.targetName() + ")";
            log.info("Configured {} DNN backend={} target={} [validated]", modelName, resolved.backendName(), resolved.targetName());
        } catch (Exception e) {
            candidate.deallocate();
            if (!fallbackToCpu) {
                throw new IllegalStateException(
                        "Failed to configure " + modelName + " with backend=" + resolved.backendName()
                                + " target=" + resolved.targetName(), e);
            }
            log.warn("{} acceleration (backend={}, target={}) failed validation, falling back to CPU: {}",
                    modelName, resolved.backendName(), resolved.targetName(), e.getMessage());
            Net cpuNet = loadNet(modelPath);
            cpuNet.setPreferableBackend(DNN_BACKEND_OPENCV);
            cpuNet.setPreferableTarget(DNN_TARGET_CPU);
            createdNet = cpuNet;
            createdDescription = "OpenCV DNN (backend=OPENCV, target=CPU) [fallback]";
        }
        this.net = createdNet;
        this.description = createdDescription;
    }

    @Override
    public TensorData forward(Mat blob) {
        net.setInput(blob);
        Mat out = net.forward();
        try {
            return toTensorData(out);
        } finally {
            out.deallocate();
        }
    }

    @Override
    public Map<String, TensorData> forward(Mat blob, List<String> outputNames) {
        net.setInput(blob);
        StringVector names = new StringVector(outputNames.toArray(new String[0]));
        MatVector out = new MatVector(outputNames.size());
        try {
            net.forward(out, names);
            Map<String, TensorData> result = new HashMap<>();
            for (long i = 0; i < out.size(); i++) {
                Mat m = out.get(i);
                result.put(sanitize(outputNames.get((int) i)), toTensorData(m));
                m.deallocate();
            }
            return result;
        } finally {
            out.deallocate();
            names.deallocate();
        }
    }

    @Override
    public List<String> getOutputNames() {
        StringVector names = net.getUnconnectedOutLayersNames();
        try {
            List<String> result = new ArrayList<>();
            for (long i = 0; i < names.size(); i++) {
                result.add(sanitize(names.get(i).getString()));
            }
            return result;
        } finally {
            names.deallocate();
        }
    }

    @Override
    public String describe() {
        return description;
    }

    @Override
    public void close() {
        net.deallocate();
    }

    // ------------------------------------------------------------------
    // Backend / target resolution
    // ------------------------------------------------------------------

    private record BackendTarget(int backend, int target, String backendName, String targetName) {}

    private static BackendTarget resolveBackendTarget(
            AccelerationProperties.Backend backend,
            AccelerationProperties.Target target
    ) {
        if (backend == null || backend == AccelerationProperties.Backend.AUTO) {
            return detectBest();
        }
        int cvBackend = mapBackend(backend);
        AccelerationProperties.Target effectiveTarget = (target == null || target == AccelerationProperties.Target.AUTO)
                ? defaultTargetFor(backend)
                : target;
        return new BackendTarget(cvBackend, mapTarget(effectiveTarget), backend.name(), effectiveTarget.name());
    }

    /**
     * Auto-detection: CUDA &gt; OpenCL &gt; Vulkan &gt; CPU, based on the backends/targets
     * compiled into the running OpenCV build.
     */
    private static BackendTarget detectBest() {
        IntIntPairVector available = getAvailableBackends();
        try {
            for (long i = 0; i < available.size(); i++) {
                int b = available.first(i);
                int t = available.second(i);
                if (b == DNN_BACKEND_CUDA && (t == DNN_TARGET_CUDA || t == DNN_TARGET_CUDA_FP16)) {
                    return new BackendTarget(DNN_BACKEND_CUDA, DNN_TARGET_CUDA, "CUDA", "CUDA");
                }
            }
            for (long i = 0; i < available.size(); i++) {
                int b = available.first(i);
                int t = available.second(i);
                if (b == DNN_BACKEND_OPENCV && t == DNN_TARGET_OPENCL) {
                    return new BackendTarget(DNN_BACKEND_OPENCV, DNN_TARGET_OPENCL, "OPENCV", "OPENCL");
                }
            }
            for (long i = 0; i < available.size(); i++) {
                int b = available.first(i);
                int t = available.second(i);
                if (b == DNN_BACKEND_VKCOM && t == DNN_TARGET_VULKAN) {
                    return new BackendTarget(DNN_BACKEND_VKCOM, DNN_TARGET_VULKAN, "VKCOM", "VULKAN");
                }
            }
            return new BackendTarget(DNN_BACKEND_OPENCV, DNN_TARGET_CPU, "OPENCV", "CPU");
        } finally {
            available.deallocate();
        }
    }

    private static int mapBackend(AccelerationProperties.Backend backend) {
        return switch (backend) {
            case AUTO, OPENCV, OPENCL -> DNN_BACKEND_OPENCV;
            case CUDA -> DNN_BACKEND_CUDA;
            case VULKAN -> DNN_BACKEND_VKCOM;
            case ONNX_CPU, ONNX_CUDA, ONNX_ROCM ->
                    throw new IllegalArgumentException("ONNX backends are handled by OnnxRuntimeEngine, not OpenCvDnnEngine");
        };
    }

    private static int mapTarget(AccelerationProperties.Target target) {
        return switch (target) {
            case AUTO, CPU -> DNN_TARGET_CPU;
            case OPENCL -> DNN_TARGET_OPENCL;
            case OPENCL_FP16 -> DNN_TARGET_OPENCL_FP16;
            case CUDA -> DNN_TARGET_CUDA;
            case CUDA_FP16 -> DNN_TARGET_CUDA_FP16;
            case VULKAN -> DNN_TARGET_VULKAN;
        };
    }

    private static AccelerationProperties.Target defaultTargetFor(AccelerationProperties.Backend backend) {
        return switch (backend) {
            case CUDA -> AccelerationProperties.Target.CUDA;
            case OPENCL -> AccelerationProperties.Target.OPENCL;
            case VULKAN -> AccelerationProperties.Target.VULKAN;
            default -> AccelerationProperties.Target.CPU;
        };
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    private static Net loadNet(String modelPath) {
        Net net = readNetFromONNX(modelPath);
        if (net == null || net.empty()) {
            if (net != null) {
                net.deallocate();
            }
            throw new IllegalStateException("Failed to load network from " + modelPath);
        }
        return net;
    }

    /**
     * Single dummy forward pass to prove the configured backend actually works.
     * This is the watchdog: it runs once at construction, never in a loop.
     */
    private static void validateForward(Net net, String modelName, BackendTarget resolved) {
        Mat dummyImage = null, blob = null, out = null;
        Size size = null;
        Scalar mean = null;
        try {
            dummyImage = new Mat(TEST_INPUT_SIZE, TEST_INPUT_SIZE, CV_32FC3, new Scalar(0, 0, 0, 0));
            size = new Size(TEST_INPUT_SIZE, TEST_INPUT_SIZE);
            mean = new Scalar(0, 0, 0, 0);
            blob = blobFromImage(dummyImage, 1.0 / 255.0, size, mean, true, false, CV_32F);
            net.setInput(blob);
            out = net.forward();
            if (out == null || out.empty()) {
                throw new IllegalStateException("validation forward returned empty output");
            }
        } finally {
            if (out != null) {
                out.deallocate();
            }
            if (blob != null) {
                blob.deallocate();
            }
            if (dummyImage != null) {
                dummyImage.deallocate();
            }
            if (size != null) {
                size.deallocate();
            }
            if (mean != null) {
                mean.deallocate();
            }
        }
    }

    private static TensorData toTensorData(Mat m) {
        long[] shape = new long[m.dims()];
        for (int i = 0; i < m.dims(); i++) {
            shape[i] = m.size(i);
        }
        float[] data = new float[(int) m.total()];
        FloatPointer pointer = new FloatPointer(m.data());
        pointer.limit(data.length);
        pointer.get(data);
        return new TensorData(data, shape);
    }

    private static String sanitize(String outputName) {
        if (outputName == null) {
            return "";
        }
        return outputName.replace("\u0000", "").trim();
    }
}