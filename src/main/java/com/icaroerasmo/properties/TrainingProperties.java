package com.icaroerasmo.properties;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

@Getter
@Setter
@Component
@ConfigurationProperties(prefix = "object-detection.training")
public class TrainingProperties {
    /**
     * Classpath folder that contains the training dataset images.
     * Default: "training" under src/main/resources.
     */
    private String rootFolder = "train";
}
