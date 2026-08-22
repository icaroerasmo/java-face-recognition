package com.icaroerasmo.detectors.shared;

/**
 * Pure PASCAL VOC class filter shared by {@code PersonDetector} and {@code PetDetector}.
 * Kept free of OpenCV so the classification logic is unit-testable without native
 * resources.
 *
 * <p>The shared {@code SSD_MobileNet} model is trained on PASCAL VOC (prototxt
 * {@code num_classes: 21}, i.e. 20 VOC classes + background), so class ids are the
 * 1-indexed VOC ids (0 = background): 7 car, 8 cat, 12 dog, 15 person, 16 pottedplant.
 */
public final class DetectionClassFilter {

    // PASCAL VOC class ids (1-indexed, 0 = background)
    public static final int PERSON_CLASS_ID = 15;
    public static final int CAR_CLASS_ID = 7;
    public static final int DOG_CLASS_ID = 12;
    public static final int CAT_CLASS_ID = 8;
    public static final int POTTED_PLANT_CLASS_ID = 16;

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

    /**
     * Intersection-over-union of two axis-aligned boxes given by integer coordinates.
     * Mirrors the person-detector suppression math exactly (including the
     * {@code ix2 <= ix1 || iy2 <= iy1 -> 0.0} edge-touching rule and the
     * {@code union <= 0 -> 0.0} zero-area rule), but is pure and testable without
     * native OpenCV.
     */
    public static double intersectionOverUnion(
            int ax, int ay, int aw, int ah,
            int bx, int by, int bw, int bh
    ) {
        int ix1 = Math.max(ax, bx);
        int iy1 = Math.max(ay, by);
        int ix2 = Math.min(ax + aw, bx + bw);
        int iy2 = Math.min(ay + ah, by + bh);
        if (ix2 <= ix1 || iy2 <= iy1) {
            return 0.0;
        }
        double inter = (double) (ix2 - ix1) * (iy2 - iy1);
        double union = (double) aw * ah + (double) bw * bh - inter;
        return union <= 0 ? 0.0 : inter / union;
    }

    /**
     * Strict suppression decision: a box is suppressed when its IoU with an
     * overlapping box strictly exceeds the suppression threshold (mirrors the
     * person detector's {@code >} comparison).
     */
    public static boolean shouldSuppress(double iou, double suppressionIouThreshold) {
        return iou > suppressionIouThreshold;
    }

    /**
     * Human-readable PASCAL VOC class token used for diagnostic logging. Known classes
     * are named; any other class falls back to its raw class id (e.g. {@code class3}).
     */
    public static String classIdToName(int classId) {
        return switch (classId) {
            case PERSON_CLASS_ID -> "person";
            case CAR_CLASS_ID -> "car";
            case DOG_CLASS_ID -> "dog";
            case CAT_CLASS_ID -> "cat";
            case POTTED_PLANT_CLASS_ID -> "pottedplant";
            default -> "class" + classId;
        };
    }
}
