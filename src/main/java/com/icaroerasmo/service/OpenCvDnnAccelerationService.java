package com.icaroerasmo.service;

import com.icaroerasmo.properties.AccelerationProperties;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.bytedeco.opencv.opencv_dnn.Net;
import org.springframework.stereotype.Service;

import java.util.Locale;

import static org.bytedeco.opencv.global.opencv_core.getBuildInformation;
import static org.bytedeco.opencv.global.opencv_core.haveOpenCL;
import static org.bytedeco.opencv.global.opencv_core.setUseOpenCL;
import static org.bytedeco.opencv.global.opencv_core.useOpenCL;
import static org.bytedeco.opencv.global.opencv_dnn.DNN_BACKEND_CUDA;
import static org.bytedeco.opencv.global.opencv_dnn.DNN_BACKEND_DEFAULT;
import static org.bytedeco.opencv.global.opencv_dnn.DNN_BACKEND_OPENCV;
import static org.bytedeco.opencv.global.opencv_dnn.DNN_TARGET_CPU;
import static org.bytedeco.opencv.global.opencv_dnn.DNN_TARGET_CUDA;
import static org.bytedeco.opencv.global.opencv_dnn.DNN_TARGET_CUDA_FP16;
import static org.bytedeco.opencv.global.opencv_dnn.DNN_TARGET_OPENCL;
import static org.bytedeco.opencv.global.opencv_dnn.DNN_TARGET_OPENCL_FP16;

@Log4j2
@Service
@RequiredArgsConstructor
public class OpenCvDnnAccelerationService {

    private final AccelerationProperties accelerationProperties;

    private volatile boolean openclAvailable;
    private volatile boolean cudaSupported;

    @PostConstruct
    void initialize() {
        String buildInfo = getBuildInformation().getString();
        cudaSupported = hasCudaSupport(buildInfo);

        boolean shouldEnableOpencl = accelerationProperties.isEnableOpencl();
        if (shouldEnableOpencl) {
            setUseOpenCL(true);
        } else {
            setUseOpenCL(false);
        }

        openclAvailable = shouldEnableOpencl && haveOpenCL() && useOpenCL();

        log.info(
            "OpenCV acceleration initialized: requested backend={}, requested target={}, openclEnabled={}, openclAvailable={}, cudaSupported={}, fallbackToCpu={}",
            accelerationProperties.getBackend(),
            accelerationProperties.getTarget(),
            shouldEnableOpencl,
            openclAvailable,
            cudaSupported,
            accelerationProperties.isFallbackToCpu()
        );
    }

    public void configure(Net net, String modelName) {
        AccelerationProperties.Backend backend = resolveBackend();
        AccelerationProperties.Target target = resolveTarget(backend, modelName);

        try {
            net.setPreferableBackend(toBackendConstant(backend));
            net.setPreferableTarget(toTargetConstant(target));
            log.info("Configured {} net with backend={} and target={}", modelName, backend, target);
        } catch (RuntimeException e) {
            if (!accelerationProperties.isFallbackToCpu()) {
                throw e;
            }

            log.warn(
                "Falling back to CPU for {} because backend={} target={} could not be applied: {}",
                modelName,
                backend,
                target,
                e.getMessage()
            );
            net.setPreferableBackend(DNN_BACKEND_OPENCV);
            net.setPreferableTarget(DNN_TARGET_CPU);
        }
    }

    private AccelerationProperties.Backend resolveBackend() {
        AccelerationProperties.Backend configured = accelerationProperties.getBackend();
        if (configured == null || configured == AccelerationProperties.Backend.AUTO) {
            if (cudaSupported) {
                return AccelerationProperties.Backend.CUDA;
            }
            if (openclAvailable) {
                return AccelerationProperties.Backend.OPENCV;
            }
            return AccelerationProperties.Backend.OPENCV;
        }

        if (configured == AccelerationProperties.Backend.CUDA && !cudaSupported) {
            return fallbackBackend("CUDA backend requested but this OpenCV build has no CUDA support");
        }

        return configured;
    }

    private AccelerationProperties.Target resolveTarget(AccelerationProperties.Backend backend, String modelName) {
        AccelerationProperties.Target configured = resolveConfiguredTarget(modelName);
        if (configured == null || configured == AccelerationProperties.Target.AUTO) {
            if (backend == AccelerationProperties.Backend.CUDA && cudaSupported) {
                return AccelerationProperties.Target.CUDA_FP16;
            }
            if (openclAvailable) {
                return AccelerationProperties.Target.OPENCL;
            }
            return AccelerationProperties.Target.CPU;
        }

        if ((configured == AccelerationProperties.Target.OPENCL
            || configured == AccelerationProperties.Target.OPENCL_FP16) && !openclAvailable) {
            return fallbackTarget("OpenCL target requested but OpenCL is unavailable at runtime");
        }

        if ((configured == AccelerationProperties.Target.CUDA
            || configured == AccelerationProperties.Target.CUDA_FP16) && !cudaSupported) {
            return fallbackTarget("CUDA target requested but this OpenCV build has no CUDA support");
        }

        return configured;
    }

    private AccelerationProperties.Target resolveConfiguredTarget(String modelName) {
        if (modelName == null) {
            return accelerationProperties.getTarget();
        }

        String normalizedModelName = modelName.toLowerCase(Locale.ROOT);
        if (normalizedModelName.contains("person")) {
            return accelerationProperties.getPersonDetectionTarget();
        }
        if (normalizedModelName.contains("face")) {
            return accelerationProperties.getFaceDetectionTarget();
        }
        return accelerationProperties.getTarget();
    }

    private AccelerationProperties.Backend fallbackBackend(String reason) {
        if (!accelerationProperties.isFallbackToCpu()) {
            throw new IllegalStateException(reason);
        }

        log.warn("{}; falling back to OpenCV backend", reason);
        return AccelerationProperties.Backend.OPENCV;
    }

    private AccelerationProperties.Target fallbackTarget(String reason) {
        if (!accelerationProperties.isFallbackToCpu()) {
            throw new IllegalStateException(reason);
        }

        log.warn("{}; falling back to CPU target", reason);
        return AccelerationProperties.Target.CPU;
    }

    private int toBackendConstant(AccelerationProperties.Backend backend) {
        return switch (backend) {
            case DEFAULT -> DNN_BACKEND_DEFAULT;
            case CUDA -> DNN_BACKEND_CUDA;
            case OPENCV, AUTO -> DNN_BACKEND_OPENCV;
        };
    }

    private int toTargetConstant(AccelerationProperties.Target target) {
        return switch (target) {
            case OPENCL -> DNN_TARGET_OPENCL;
            case OPENCL_FP16 -> DNN_TARGET_OPENCL_FP16;
            case CUDA -> DNN_TARGET_CUDA;
            case CUDA_FP16 -> DNN_TARGET_CUDA_FP16;
            case CPU, AUTO -> DNN_TARGET_CPU;
        };
    }

    private boolean hasCudaSupport(String buildInfo) {
        String normalized = buildInfo.toLowerCase(Locale.ROOT);

        for (String line : normalized.split("\\R")) {
            String trimmed = line.trim();
            if (trimmed.startsWith("nvidia cuda:")) {
                return trimmed.endsWith("yes");
            }
        }

        return !normalized.contains("unavailable:")
            || !normalized.substring(normalized.indexOf("unavailable:")).contains("cuda");
    }
}
