package com.icaroerasmo.properties;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class DetectionProperties {

    /**
     * Minimum confidence required for SSD MobileNet person detections.
     * Higher values reduce false positives such as cars being classified as people.
     */
    private double personConfidenceThreshold = 0.8;
}
