package com.icaroerasmo.properties;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

@Getter
@Setter
@Component
@ConfigurationProperties(prefix = "face-recognition.acceleration")
public class AccelerationProperties {

    private Backend backend = Backend.AUTO;
    private Target target = Target.AUTO;
    private boolean fallbackToCpu = true;
    private boolean enableOpencl = true;

    public enum Backend {
        AUTO,
        DEFAULT,
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
