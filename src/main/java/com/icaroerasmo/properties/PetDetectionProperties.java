package com.icaroerasmo.properties;

import lombok.Getter;
import lombok.Setter;

/**
 * Configuration for pet (dog/cat) detection.
 */
@Getter
@Setter
public class PetDetectionProperties {

    /**
     * Master flag for pet detection (independent of {@code object-detection.enabled}).
     */
    private boolean enabled = true;

    /**
     * Minimum confidence required for a dog/cat detection.
     */
    private double confidenceThreshold = 0.5;

    /**
     * Debounce window (ms) for overlay {@code PET_DETECTED} events per camera.
     */
    private long debounceMs = 5000;

    /**
     * Minimum interval (ms) between Telegram pet photos per camera.
     */
    private long telegramThrottleMs = 30000;
}
