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
@ConfigurationProperties(prefix = "face-recognition.streams")
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
    private int frameQueueCapacity = 2;

    /**
     * Maximum number of frames retained in a single tracking session before it is finalized.
     * This caps memory growth for long-lived tracks.
     */
    private int trackingMaxFrames = 30;

    /**
     * Maximum number of consecutive null image reads before the extractor reconnects the stream.
     * Higher values tolerate brief RTSP stalls on heavier streams.
     */
    private int maxConsecutiveNullFrames = 300;
}
