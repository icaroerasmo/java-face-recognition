package com.icaroerasmo.detectors.pet;

import com.icaroerasmo.detectors.shared.CocoDetection;
import com.icaroerasmo.detectors.shared.DetectionClassFilter;
import com.icaroerasmo.detectors.shared.MobileNetSsdDetector;
import com.icaroerasmo.properties.ObjectDetectionProperties;
import lombok.extern.log4j.Log4j2;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;

import static com.icaroerasmo.detectors.shared.DetectionClassFilter.DOG_CLASS_ID;
import static com.icaroerasmo.detectors.shared.DetectionClassFilter.POTTED_PLANT_CLASS_ID;

/**
 * Detects pets (PASCAL VOC {@code dog} = 12 and {@code cat} = 8) using the SAME
 * MobileNet-SSD model as {@code PersonDetector} (see {@link MobileNetSsdDetector}).
 *
 * <p>Mirroring the car-suppression in {@code PersonDetector}, any dog/cat box whose
 * IoU with a {@code potted plant} (PASCAL VOC class 16) box exceeds
 * {@code plantSuppressionIou} is suppressed - the model occasionally misclassifies a
 * potted plant as a dog, so an overlapping plant box vetoes the pet detection.
 *
 * <p>Synchronized per camera because the underlying {@code Net} is shared; every
 * forward pass runs inside {@link com.icaroerasmo.detectors.person.helper.DnnInferenceCoordinatorHelper#runExclusive}.
 * The returned {@link Rect}s are owned by the caller; every non-returned rect
 * (including plant boxes and suppressed pet boxes) is deallocated exactly once in
 * the {@code finally} block.
 */
@Log4j2
@Service
public class PetDetector {

    private final MobileNetSsdDetector mobileNetSsdDetector;
    private final double petConfidenceThreshold;
    private final double plantConfidenceThreshold;
    private final double plantSuppressionIou;

    public PetDetector(
            MobileNetSsdDetector mobileNetSsdDetector,
            ObjectDetectionProperties objectDetectionProperties
    ) {
        this.mobileNetSsdDetector = mobileNetSsdDetector;
        var pet = objectDetectionProperties.getDetection().getPet();
        this.petConfidenceThreshold = clamp01(pet.getConfidenceThreshold());
        this.plantConfidenceThreshold = clamp01(pet.getPlantConfidenceThreshold());
        this.plantSuppressionIou = clamp01(pet.getPlantSuppressionIou());
    }

    public synchronized List<PetDetection> detect(Mat image) {
        List<CocoDetection> detections = mobileNetSsdDetector.detectRaw(image);
        if (detections.isEmpty()) {
            return List.of();
        }

        List<CocoDetection> petCandidates = new ArrayList<>();
        List<CocoDetection> plants = new ArrayList<>();
        // Rects transferred to the returned list must NOT be released here.
        Set<Rect> returned = Collections.newSetFromMap(new IdentityHashMap<>());

        try {
            for (CocoDetection detection : detections) {
                if (DetectionClassFilter.classify(
                        detection.classId(), detection.confidence(), 0, 0, petConfidenceThreshold)
                        == DetectionClassFilter.DetectionType.PET) {
                    petCandidates.add(detection);
                } else if (detection.classId() == POTTED_PLANT_CLASS_ID
                        && detection.confidence() > plantConfidenceThreshold) {
                    // Plant boxes are never returned; they only veto pet false positives.
                    plants.add(detection);
                }
            }

            if (petCandidates.isEmpty()) {
                return List.of();
            }

            List<PetDetection> pets = new ArrayList<>(petCandidates.size());
            for (CocoDetection candidate : petCandidates) {
                Rect petRect = candidate.rect();
                boolean suppressed = false;
                for (CocoDetection plant : plants) {
                    Rect plantRect = plant.rect();
                    if (DetectionClassFilter.shouldSuppress(
                            DetectionClassFilter.intersectionOverUnion(
                                    petRect.x(), petRect.y(), petRect.width(), petRect.height(),
                                    plantRect.x(), plantRect.y(), plantRect.width(), plantRect.height()),
                            plantSuppressionIou)) {
                        suppressed = true;
                        break;
                    }
                }
                if (suppressed) {
                    log.debug("Suppressing pet box at ({}, {}) size {}x{} (overlaps potted plant with IoU > {})",
                            petRect.x(), petRect.y(), petRect.width(), petRect.height(),
                            String.format("%.2f", plantSuppressionIou));
                } else {
                    String label = candidate.classId() == DOG_CLASS_ID ? "Dog" : "Cat";
                    pets.add(new PetDetection(label, petRect));
                    returned.add(petRect);
                }
            }
            return pets;
        } catch (Exception e) {
            log.error("Error during pet detection: {}", e.getMessage(), e);
            // No rects may leave this method; clear the returned-set so the finally block
            // deallocates every detection rect exactly once.
            returned.clear();
            return List.of();
        } finally {
            for (CocoDetection detection : detections) {
                Rect rect = detection.rect();
                if (rect != null && !returned.contains(rect)) {
                    rect.deallocate();
                }
            }
        }
    }

    private static double clamp01(double value) {
        return Math.max(0.0, Math.min(1.0, value));
    }
}
