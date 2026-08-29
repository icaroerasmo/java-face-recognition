package com.icaroerasmo.detectors.shared.engine;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtEpDevice;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtProvider;
import ai.onnxruntime.OrtSession;
import com.icaroerasmo.properties.AccelerationProperties;
import lombok.extern.log4j.Log4j2;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.opencv.opencv_core.Mat;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * ONNX Runtime engine. Supports the CPU EP out of the box; the CUDA EP is used when
 * the {@code onnxruntime_gpu} artifact plus CUDA 12 / cuDNN 9 are present on the host.
 *
 * <p>Watchdog: the preferred execution provider is attempted exactly once at
 * construction. If session creation fails and {@code fallbackToCpu} is enabled, the
 * session is recreated on CPU exactly once - never more. There is no retry loop.
 *
 * <p>ROCm: the ROCm EP was removed from ONNX Runtime 1.23+ and no Java artifact ships
 * it. {@code ONNX_ROCM} is accepted for config compatibility: if a ROCm device is
 * reported by the runtime it is used, otherwise the engine falls back to CPU with a
 * clear warning.
 */
@Log4j2
public class OnnxRuntimeEngine implements DnnEngine {

    private final OrtEnvironment env;
    private final OrtSession session;
    private final String inputName;
    private final List<String> outputNames;
    private final String description;

    public OnnxRuntimeEngine(
            String modelPath,
            AccelerationProperties.Backend backend,
            boolean fallbackToCpu,
            String modelName
    ) {
        this.env = OrtEnvironment.getEnvironment();
        OrtSession createdSession;
        String createdDescription;
        String ep = "CPU";
        try {
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            if (backend == AccelerationProperties.Backend.ONNX_CUDA && hasProvider(OrtProvider.CUDA)) {
                options.addCUDA(0);
                ep = "CUDA";
            } else if (backend == AccelerationProperties.Backend.ONNX_ROCM) {
                OrtEpDevice rocmDevice = findRocmDevice();
                if (rocmDevice != null) {
                    options.addExecutionProvider(Collections.singletonList(rocmDevice), Collections.emptyMap());
                    ep = "ROCm";
                } else {
                    log.warn("{}: ROCm EP not available (removed from ONNX Runtime 1.23+; no Java artifact ships it). Using CPU.", modelName);
                }
            } else if (backend != AccelerationProperties.Backend.ONNX_CPU) {
                log.warn("{}: execution provider for {} not available (providers: {}). Using CPU.",
                        modelName, backend, OrtEnvironment.getAvailableProviders());
            }
            options.addCPU(true);
            createdSession = env.createSession(modelPath, options);
            createdDescription = "ONNX Runtime (EP=" + ep + ")";
            log.info("Configured {} ONNX Runtime EP={}", modelName, ep);
        } catch (Exception e) {
            if (!fallbackToCpu) {
                throw new IllegalStateException("Failed to create ONNX Runtime session with EP=" + ep + " for " + modelName, e);
            }
            log.warn("{}: failed to create ONNX Runtime session with EP={}, falling back to CPU: {}",
                    modelName, ep, e.getMessage());
            OrtSession.SessionOptions cpuOptions = new OrtSession.SessionOptions();
            try {
                cpuOptions.addCPU(true);
                createdSession = env.createSession(modelPath, cpuOptions);
            } catch (OrtException ortException) {
                throw new IllegalStateException("Failed to create ONNX Runtime session even on CPU for " + modelName, ortException);
            }
            createdDescription = "ONNX Runtime (EP=CPU) [fallback]";
        }
        this.session = createdSession;
        this.description = createdDescription;
        Set<String> inputs = session.getInputNames();
        this.inputName = inputs.isEmpty() ? null : inputs.iterator().next();
        this.outputNames = new ArrayList<>(session.getOutputNames());
    }

    @Override
    public TensorData forward(Mat blob) {
        try (OnnxTensor input = toOnnxTensor(blob);
             OrtSession.Result result = session.run(Collections.singletonMap(inputName, input))) {
            return toTensorData((OnnxTensor) result.get(0));
        } catch (OrtException e) {
            throw new IllegalStateException("ONNX Runtime forward pass failed", e);
        }
    }

    @Override
    public Map<String, TensorData> forward(Mat blob, List<String> outputNames) {
        try (OnnxTensor input = toOnnxTensor(blob);
             OrtSession.Result result = session.run(
                     Collections.singletonMap(inputName, input),
                     Set.copyOf(outputNames))) {
            Map<String, TensorData> map = new HashMap<>();
            for (String name : outputNames) {
                map.put(name, toTensorData((OnnxTensor) result.get(name).orElseThrow(
                        () -> new IllegalStateException("ONNX Runtime did not return output '" + name + "'"))));
            }
            return map;
        } catch (OrtException e) {
            throw new IllegalStateException("ONNX Runtime multi-output forward pass failed", e);
        }
    }

    @Override
    public List<String> getOutputNames() {
        return new ArrayList<>(outputNames);
    }

    @Override
    public String describe() {
        return description;
    }

    @Override
    public void close() {
        try {
            session.close();
        } catch (Exception e) {
            log.warn("Failed to close ONNX Runtime session: {}", e.getMessage());
        }
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    private OnnxTensor toOnnxTensor(Mat blob) {
        float[] data = new float[(int) blob.total()];
        FloatPointer pointer = new FloatPointer(blob.data());
        pointer.limit(data.length);
        pointer.get(data);
        long[] shape = new long[blob.dims()];
        for (int i = 0; i < blob.dims(); i++) {
            shape[i] = blob.size(i);
        }
        try {
            return OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape);
        } catch (OrtException e) {
            throw new IllegalStateException("Failed to create ONNX Runtime input tensor", e);
        }
    }

    private static TensorData toTensorData(OnnxTensor tensor) {
        long[] shape = tensor.getInfo().getShape();
        FloatBuffer buffer = tensor.getFloatBuffer();
        float[] data = new float[buffer.remaining()];
        buffer.get(data);
        return new TensorData(data, shape);
    }

    private static boolean hasProvider(OrtProvider provider) {
        return OrtEnvironment.getAvailableProviders().contains(provider);
    }

    private static OrtEpDevice findRocmDevice() {
        try {
            for (OrtEpDevice device : OrtEnvironment.getEnvironment().getEpDevices()) {
                if ("ROCM".equalsIgnoreCase(device.getEpName())) {
                    return device;
                }
            }
        } catch (OrtException e) {
            log.warn("Failed to enumerate ONNX Runtime EP devices: {}", e.getMessage());
        }
        return null;
    }
}