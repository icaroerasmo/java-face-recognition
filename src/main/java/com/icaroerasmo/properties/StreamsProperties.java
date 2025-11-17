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
    private List<Camera> cameras = new ArrayList<>();
}
