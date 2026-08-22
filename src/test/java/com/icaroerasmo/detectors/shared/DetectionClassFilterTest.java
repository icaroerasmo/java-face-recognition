package com.icaroerasmo.detectors.shared;

import org.junit.jupiter.api.Test;

import static com.icaroerasmo.detectors.shared.DetectionClassFilter.CAR_CLASS_ID;
import static com.icaroerasmo.detectors.shared.DetectionClassFilter.CAT_CLASS_ID;
import static com.icaroerasmo.detectors.shared.DetectionClassFilter.DOG_CLASS_ID;
import static com.icaroerasmo.detectors.shared.DetectionClassFilter.PERSON_CLASS_ID;
import static com.icaroerasmo.detectors.shared.DetectionClassFilter.DetectionType.CAR;
import static com.icaroerasmo.detectors.shared.DetectionClassFilter.DetectionType.NONE;
import static com.icaroerasmo.detectors.shared.DetectionClassFilter.DetectionType.PERSON;
import static com.icaroerasmo.detectors.shared.DetectionClassFilter.DetectionType.PET;
import static com.icaroerasmo.detectors.shared.DetectionClassFilter.classify;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Pure tests for the shared COCO class filter used by PersonDetector and PetDetector.
 */
class DetectionClassFilterTest {

    @Test
    void personAboveThresholdIsPerson() {
        assertEquals(PERSON, classify(PERSON_CLASS_ID, 0.85f, 0.8, 0.5, 0.5));
    }

    @Test
    void personAtThresholdIsNotPerson() {
        // Strict (exclusive) comparison like the original person detector.
        // 0.8 is not exactly representable as a float, so use a clearly below value.
        assertEquals(NONE, classify(PERSON_CLASS_ID, 0.7999f, 0.8, 0.5, 0.5));
    }

    @Test
    void personBelowThresholdIsNone() {
        assertEquals(NONE, classify(PERSON_CLASS_ID, 0.3f, 0.8, 0.5, 0.5));
    }

    @Test
    void dogAboveThresholdIsPet() {
        assertEquals(PET, classify(DOG_CLASS_ID, 0.7f, 0.8, 0.5, 0.5));
    }

    @Test
    void catAboveThresholdIsPet() {
        assertEquals(PET, classify(CAT_CLASS_ID, 0.9f, 0.8, 0.5, 0.5));
    }

    @Test
    void petAtThresholdIsNotPet() {
        assertEquals(NONE, classify(DOG_CLASS_ID, 0.5f, 0.8, 0.5, 0.5));
    }

    @Test
    void petBelowThresholdIsNone() {
        assertEquals(NONE, classify(DOG_CLASS_ID, 0.4f, 0.8, 0.5, 0.5));
        assertEquals(NONE, classify(CAT_CLASS_ID, 0.49f, 0.8, 0.5, 0.5));
    }

    @Test
    void carAboveThresholdIsCar() {
        assertEquals(CAR, classify(CAR_CLASS_ID, 0.6f, 0.8, 0.5, 0.5));
    }

    @Test
    void carBelowThresholdIsNone() {
        assertEquals(NONE, classify(CAR_CLASS_ID, 0.2f, 0.8, 0.5, 0.5));
    }

    @Test
    void unrelatedClassIsNone() {
        assertEquals(NONE, classify(3, 0.99f, 0.8, 0.5, 0.5));
    }

    @Test
    void personIsNotConfusedWithPetEvenWithPetThresholdZero() {
        // A person must never classify as PET regardless of thresholds.
        assertEquals(PERSON, classify(PERSON_CLASS_ID, 0.9f, 0.5, 0.5, 0.0));
    }
}
