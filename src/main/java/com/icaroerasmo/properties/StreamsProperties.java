package com.icaroerasmo.properties;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;

@Getter
@Setter
@Component
@ConfigurationProperties(prefix = "object-detection.streams")
public class StreamsProperties {

    /**
     * Camera configurations with name and RTSP URL
     */
    private List<CameraProperties> cameras = new ArrayList<>();

    /**
     * Number of processed frames per second.
     * Lower values reduce latency and memory pressure by skipping stale frames.
     */
    private int processingFps = 5;

    /**
     * Number of queued decoded frames per camera.
     * Keep this low to prefer freshness over backlog.
     */
    private int frameQueueCapacity = 10;

    /**
     * Optional downscale width applied to frames before they are queued for
     * processing. 0 = keep the native stream resolution. A value like 960 or 640
     * reduces memory and speeds up the detection pipeline at the cost of some
     * detail (mainly affecting recognition of small/distant faces).
     */
    private int processingWidth = 0;

    /**
     * Rate (frames per second) at which the cheap frame-differencing movement
     * detection runs in the producer, decoupled from the slower DNN
     * {@link #processingFps}. Higher values give faster movement triggers.
     */
    private int movementFps = 10;

    /**
     * Maximum number of frames retained in a single tracking session before it is finalized.
     * This caps memory growth for long-lived tracks.
     */
    private int trackingMaxFrames = 30;

    /**
     * Minimum number of frames a known identity must be observed before it is
     * confirmed as that person. 1 = recognize on the first frame (fastest);
     * raise this if identity flicker becomes a problem.
     */
    private int identityMinFrames = 1;

    /**
     * Maximum number of consecutive null image reads before the extractor reconnects the stream.
     * Higher values tolerate brief RTSP stalls on heavier streams.
     */
    private int maxConsecutiveNullFrames = 300;
}
