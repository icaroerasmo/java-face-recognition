package com.icaroerasmo.properties;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.NestedConfigurationProperty;
import org.springframework.stereotype.Component;

@Getter
@Setter
@Component
@ConfigurationProperties(prefix = "face-recognition")
public class FaceRecognitionProperties {

    @NestedConfigurationProperty
    private StreamsProperties streams = new StreamsProperties();

    @NestedConfigurationProperty
    private TrainingProperties training = new TrainingProperties();

    @NestedConfigurationProperty
    private TelegramProperties telegram = new TelegramProperties();

    @NestedConfigurationProperty
    private AccelerationProperties acceleration = new AccelerationProperties();
}
