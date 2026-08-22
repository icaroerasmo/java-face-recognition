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

import static com.icaroerasmo.detectors.shared.DetectionClassFilter.CAT_CLASS_ID;
import static com.icaroerasmo.detectors.shared.DetectionClassFilter.DOG_CLASS_ID;

/**
 * Detects pets (COCO {@code dog} = 16 and {@code cat} = 17) using the SAME
 * MobileNet-SSD model as {@code PersonDetector} (see {@link MobileNetSsdDetector}).
 *
 * <p>Synchronized per camera because the underlying {@code Net} is shared; every
 * forward pass runs inside {@link com.icaroerasmo.detectors.person.helper.DnnInferenceCoordinatorHelper#runExclusive}.
 * The returned {@link Rect}s are owned by the caller.
 */
@Log4j2
@Service
public class PetDetector {

    private final MobileNetSsdDetector mobileNetSsdDetector;
    private final double petConfidenceThreshold;

    public PetDetector(
            MobileNetSsdDetector mobileNetSsdDetector,
            ObjectDetectionProperties objectDetectionProperties
    ) {
        this.mobileNetSsdDetector = mobileNetSsdDetector;
        this.petConfidenceThreshold = Math.max(0.0, Math.min(1.0,
                objectDetectionProperties.getDetection().getPet().getConfidenceThreshold()));
    }

    public synchronized List<PetDetection> detect(Mat image) {
        List<CocoDetection> detections = mobileNetSsdDetector.detectRaw(image);
        if (detections.isEmpty()) {
            return List.of();
        }

        List<PetDetection> pets = new ArrayList<>();
        // Rects transferred to the returned list must NOT be released here.
        Set<Rect> returned = Collections.newSetFromMap(new IdentityHashMap<>());
        try {
            for (CocoDetection detection : detections) {
                if (DetectionClassFilter.classify(
                        detection.classId(), detection.confidence(), 0, 0, petConfidenceThreshold)
                        != DetectionClassFilter.DetectionType.PET) {
                    continue;
                }
                String label = detection.classId() == DOG_CLASS_ID ? "Dog" : "Cat";
                pets.add(new PetDetection(label, detection.rect()));
                returned.add(detection.rect());
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
}
