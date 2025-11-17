package com.icaroerasmo.properties;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;

@Getter
@Setter
@ConfigurationProperties(prefix = "face-recognition.streams")
public class StreamsProperties {

    /**
     * RTSP URL used by the recognition pipeline.
     */
    private String rtspUrl;
}

