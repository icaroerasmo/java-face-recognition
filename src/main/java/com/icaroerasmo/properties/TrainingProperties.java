package com.icaroerasmo.properties;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

@Getter
@Setter
@Component
@ConfigurationProperties(prefix = "face-recognition.training")
public class TrainingProperties {

    /**
     * Path (relative or absolute) to the trained dataset XML file.
     * Default: "trained_dataset.xml" in the working directory.
     */
    private String datasetPath = "trained_dataset.xml";

    /**
     * Classpath folder that contains the training dataset images.
     * Default: "training" under src/main/resources.
     */
    private String rootFolder = "train";
}

