package com.icaroerasmo.properties;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class AccelerationProperties {

    private Backend backend = Backend.OPENCV;
    private Target target = Target.CPU;
    private Target faceDetectionTarget = Target.OPENCL;
    private Target personDetectionTarget = Target.CPU;
    private boolean enableOpencl = true;
    private boolean fallbackToCpu = true;

    public enum Backend {
        AUTO,
        OPENCV,
        CUDA
    }

    public enum Target {
        AUTO,
        CPU,
        OPENCL,
        OPENCL_FP16,
        CUDA,
        CUDA_FP16
    }
}
