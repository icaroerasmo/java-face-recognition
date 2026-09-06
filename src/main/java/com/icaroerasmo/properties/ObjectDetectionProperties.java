package com.icaroerasmo.properties;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.NestedConfigurationProperty;

@Getter
@Setter
@ConfigurationProperties(prefix = "object-detection")
public class ObjectDetectionProperties {

    private Boolean enabled = true;

    @NestedConfigurationProperty
    private StreamsProperties streams = new StreamsProperties();

    @NestedConfigurationProperty
    private TrainingProperties training = new TrainingProperties();

    @NestedConfigurationProperty
    private ClipProperties clips = new ClipProperties();

    @NestedConfigurationProperty
    private AccelerationProperties acceleration = new AccelerationProperties();

    @NestedConfigurationProperty
    private DetectionProperties detection = new DetectionProperties();
}
