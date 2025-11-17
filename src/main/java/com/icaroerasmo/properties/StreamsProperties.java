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
     * RTSP URLs used by the recognition pipeline (one or more cameras).
     */
    private List<String> rtspUrls = new ArrayList<>();
}
