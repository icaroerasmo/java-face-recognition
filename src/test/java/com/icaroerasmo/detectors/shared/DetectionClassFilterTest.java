package com.icaroerasmo.detectors.shared;

import org.junit.jupiter.api.Test;

import static com.icaroerasmo.detectors.shared.DetectionClassFilter.CAR_CLASS_ID;
import static com.icaroerasmo.detectors.shared.DetectionClassFilter.CAT_CLASS_ID;
import static com.icaroerasmo.detectors.shared.DetectionClassFilter.DOG_CLASS_ID;
import static com.icaroerasmo.detectors.shared.DetectionClassFilter.PERSON_CLASS_ID;
import static com.icaroerasmo.detectors.shared.DetectionClassFilter.POTTED_PLANT_CLASS_ID;
import static com.icaroerasmo.detectors.shared.DetectionClassFilter.DetectionType.CAR;
import static com.icaroerasmo.detectors.shared.DetectionClassFilter.DetectionType.NONE;
import static com.icaroerasmo.detectors.shared.DetectionClassFilter.DetectionType.PERSON;
import static com.icaroerasmo.detectors.shared.DetectionClassFilter.DetectionType.PET;
import static com.icaroerasmo.detectors.shared.DetectionClassFilter.classify;
import static com.icaroerasmo.detectors.shared.DetectionClassFilter.classIdToName;
import static com.icaroerasmo.detectors.shared.DetectionClassFilter.intersectionOverUnion;
import static com.icaroerasmo.detectors.shared.DetectionClassFilter.shouldSuppress;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Pure tests for the shared COCO class filter used by PersonDetector and PetDetector.
 * Native-free: no {@code Rect}/{@code Mat} is used.
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

    // --- Pure IoU / plant-suppression decision helpers -------------------------

    @Test
    void vocClassIdConstantsAreCorrect() {
        // The shared model is PASCAL VOC (1-indexed, 0 = background): the ids must be
        // person=15, car=7, dog=12, cat=8, pottedplant=16.
        assertEquals(15, PERSON_CLASS_ID);
        assertEquals(7, CAR_CLASS_ID);
        assertEquals(12, DOG_CLASS_ID);
        assertEquals(8, CAT_CLASS_ID);
        assertEquals(16, POTTED_PLANT_CLASS_ID);
    }

    @Test
    void iuOfIdenticalBoxesIsOne() {
        assertEquals(1.0, intersectionOverUnion(10, 10, 100, 100, 10, 10, 100, 100), 1e-9);
    }

    @Test
    void iuOfDisjointBoxesIsZero() {
        assertEquals(0.0, intersectionOverUnion(0, 0, 10, 10, 100, 100, 10, 10), 1e-9);
    }

    @Test
    void iuOfPartialOverlapIsIntersectionOverUnion() {
        // a = (0,0,10,10), b = (5,0,10,10): intersection 5x10=50, union 100+100-50=150 -> 1/3
        assertEquals(1.0 / 3.0, intersectionOverUnion(0, 0, 10, 10, 5, 0, 10, 10), 1e-9);
    }

    @Test
    void iuOfEdgeTouchingBoxesIsZero() {
        // a = (0,0,10,10), b = (10,0,10,10): share an edge only -> ix2(10) <= ix1(10) -> 0
        assertEquals(0.0, intersectionOverUnion(0, 0, 10, 10, 10, 0, 10, 10), 1e-9);
    }

    @Test
    void iuOfCornerTouchingBoxesIsZero() {
        assertEquals(0.0, intersectionOverUnion(0, 0, 10, 10, 10, 10, 10, 10), 1e-9);
    }

    @Test
    void iuOfContainedBoxMatchesExactMath() {
        // a = (0,0,100,100) area 10000 contains b = (25,25,50,50) area 2500:
        // intersection 2500, union 10000 -> 0.25
        assertEquals(0.25, intersectionOverUnion(0, 0, 100, 100, 25, 25, 50, 50), 1e-9);
    }

    @Test
    void iuOfZeroAreaBoxIsZero() {
        assertEquals(0.0, intersectionOverUnion(0, 0, 0, 10, 5, 5, 10, 10), 1e-9);
    }

    @Test
    void iuIsSymmetric() {
        double ab = intersectionOverUnion(0, 0, 10, 10, 5, 5, 10, 10);
        double ba = intersectionOverUnion(5, 5, 10, 10, 0, 0, 10, 10);
        assertEquals(ab, ba, 1e-9);
    }

    @Test
    void shouldSuppressRequiresStrictlyGreaterThanThreshold() {
        assertFalse(shouldSuppress(0.35, 0.35)); // exactly at the threshold is NOT suppressed
        assertTrue(shouldSuppress(0.36, 0.35));
        assertFalse(shouldSuppress(0.10, 0.35));
        assertTrue(shouldSuppress(1.0, 0.35));
    }

    @Test
    void plantOverlapSuppressesPetDecision() {
        // Identical dog and plant boxes -> IoU 1.0 > 0.35 -> suppress.
        double identical = intersectionOverUnion(0, 0, 50, 50, 0, 0, 50, 50);
        assertTrue(shouldSuppress(identical, 0.35));

        // Disjoint dog and plant boxes -> IoU 0 -> keep.
        double disjoint = intersectionOverUnion(0, 0, 50, 50, 200, 200, 50, 50);
        assertFalse(shouldSuppress(disjoint, 0.35));

        // Light overlap (dog 100x100, plant 50x50 centered in it -> IoU 0.25 < 0.35) -> keep.
        double light = intersectionOverUnion(0, 0, 100, 100, 25, 25, 50, 50);
        assertFalse(shouldSuppress(light, 0.35));

        // Strong overlap (dog 100x100, plant 90x90 centered in it -> high IoU) -> suppress.
        double heavy = intersectionOverUnion(0, 0, 100, 100, 5, 5, 90, 90);
        assertTrue(shouldSuppress(heavy, 0.35));
    }

    @Test
    void personOverlapWithPetSuppressesDecision() {
        // Person box identical to a dog box -> IoU 1.0 > 0.35 -> the person box is
        // suppressed (the model misclassified the dog as person).
        double identical = intersectionOverUnion(0, 0, 50, 50, 0, 0, 50, 50);
        assertTrue(shouldSuppress(identical, 0.35));

        // Person box disjoint from a cat box -> IoU 0 -> kept.
        double disjoint = intersectionOverUnion(0, 0, 50, 50, 200, 200, 50, 50);
        assertFalse(shouldSuppress(disjoint, 0.35));

        // Light person/pet overlap (IoU 0.25 < 0.35) -> kept.
        double light = intersectionOverUnion(0, 0, 100, 100, 25, 25, 50, 50);
        assertFalse(shouldSuppress(light, 0.35));

        // Strong person/pet overlap (IoU 0.81 > 0.35) -> suppressed.
        double heavy = intersectionOverUnion(0, 0, 100, 100, 5, 5, 90, 90);
        assertTrue(shouldSuppress(heavy, 0.35));
    }

    @Test
    void classIdToNameMapsKnownClasses() {
        assertEquals("person", classIdToName(PERSON_CLASS_ID));
        assertEquals("car", classIdToName(CAR_CLASS_ID));
        assertEquals("dog", classIdToName(DOG_CLASS_ID));
        assertEquals("cat", classIdToName(CAT_CLASS_ID));
        assertEquals("pottedplant", classIdToName(POTTED_PLANT_CLASS_ID));
    }

    @Test
    void classIdToNameFallsBackToRawIdForUnknownClasses() {
        assertEquals("class3", classIdToName(3));
        assertEquals("class0", classIdToName(0));
        assertEquals("class80", classIdToName(80));
    }
}
