package com.icaroerasmo.properties;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.NestedConfigurationProperty;

@Getter
@Setter
public class DetectionProperties {

    /**
     * Minimum confidence required for SSD MobileNet person detections.
     * Higher values reduce false positives such as cars being classified as people.
     */
    private double personConfidenceThreshold = 0.8;

    /**
     * Minimum confidence required for SSD MobileNet car detections.
     * Car boxes above this threshold are used to suppress false-positive person detections.
     */
    private double carConfidenceThreshold = 0.5;

    /**
     * Maximum fraction of the frame area a person box may occupy.
     * Person boxes larger than this are rejected as implausibly large
     * (e.g. cars misclassified as people).
     */
    private double maxPersonAreaRatio = 0.45;

    @NestedConfigurationProperty
    private MovementDetectionProperties movement = new MovementDetectionProperties();

    @NestedConfigurationProperty
    private PetDetectionProperties pet = new PetDetectionProperties();
}
