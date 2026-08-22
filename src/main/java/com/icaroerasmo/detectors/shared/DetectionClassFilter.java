package com.icaroerasmo.detectors.shared;

/**
 * Pure COCO class filter shared by {@code PersonDetector} and {@code PetDetector}.
 * Kept free of OpenCV so the classification logic is unit-testable without native
 * resources.
 */
public final class DetectionClassFilter {

    public static final int PERSON_CLASS_ID = 15;
    public static final int CAR_CLASS_ID = 7;
    public static final int DOG_CLASS_ID = 16;
    public static final int CAT_CLASS_ID = 17;

    public enum DetectionType { PERSON, CAR, PET, NONE }

    private DetectionClassFilter() {
    }

    /**
     * Classifies a single detection using strict (exclusive) threshold comparisons,
     * matching the original person-detector behavior ({@code confidence > threshold}).
     */
    public static DetectionType classify(
            int classId,
            float confidence,
            double personConfidenceThreshold,
            double carConfidenceThreshold,
            double petConfidenceThreshold
    ) {
        if (classId == PERSON_CLASS_ID) {
            return confidence > personConfidenceThreshold ? DetectionType.PERSON : DetectionType.NONE;
        }
        if (classId == CAR_CLASS_ID) {
            return confidence > carConfidenceThreshold ? DetectionType.CAR : DetectionType.NONE;
        }
        if (classId == DOG_CLASS_ID || classId == CAT_CLASS_ID) {
            return confidence > petConfidenceThreshold ? DetectionType.PET : DetectionType.NONE;
        }
        return DetectionType.NONE;
    }
}
