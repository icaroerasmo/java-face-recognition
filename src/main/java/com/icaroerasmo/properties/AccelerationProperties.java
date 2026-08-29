package com.icaroerasmo.properties;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class AccelerationProperties {

    private Backend backend = Backend.AUTO;
    private Target target = Target.AUTO;
    private Target faceDetectionTarget = Target.AUTO;
    private Target personDetectionTarget = Target.AUTO;
    private boolean enableOpencl = true;
    private boolean fallbackToCpu = true;

    /**
     * Inference engine selection.
     * <ul>
     *   <li>{@code AUTO} - detect the best available backend at startup (CUDA &gt; OpenCL &gt; Vulkan &gt; CPU).</li>
     *   <li>{@code OPENCV} - OpenCV DNN on CPU.</li>
     *   <li>{@code CUDA} - OpenCV DNN on CUDA (NVIDIA GPU).</li>
     *   <li>{@code OPENCL} - OpenCV DNN targeting OpenCL.</li>
     *   <li>{@code VULKAN} - OpenCV DNN on the Vulkan backend (only if the OpenCV build includes VKCOM).</li>
     *   <li>{@code ONNX_CPU} - ONNX Runtime on CPU.</li>
     *   <li>{@code ONNX_CUDA} - ONNX Runtime CUDA EP (requires the {@code onnxruntime_gpu} artifact
     *       plus CUDA 12 / cuDNN 9 on the host; falls back to CPU otherwise).</li>
     *   <li>{@code ONNX_ROCM} - ONNX Runtime ROCm EP. NOTE: the ROCm EP was removed from ONNX Runtime
     *       1.23+ and no Java artifact ships it; this value is accepted for config compatibility and
     *       falls back to CPU with a warning.</li>
     * </ul>
     */
    public enum Backend {
        AUTO,
        OPENCV,
        CUDA,
        OPENCL,
        VULKAN,
        ONNX_CPU,
        ONNX_CUDA,
        ONNX_ROCM
    }

    /**
     * OpenCV DNN compute target. Only meaningful for the OpenCV-based backends
     * ({@code OPENCV}, {@code CUDA}, {@code OPENCL}, {@code VULKAN}).
     */
    public enum Target {
        AUTO,
        CPU,
        OPENCL,
        OPENCL_FP16,
        CUDA,
        CUDA_FP16,
        VULKAN
    }
}