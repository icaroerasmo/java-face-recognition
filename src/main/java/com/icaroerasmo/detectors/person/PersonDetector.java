package com.icaroerasmo.detectors.person;

import com.icaroerasmo.detectors.IDetector;
import com.icaroerasmo.detectors.shared.CocoDetection;
import com.icaroerasmo.detectors.shared.DetectionClassFilter;
import com.icaroerasmo.detectors.shared.YoloDetector;
import com.icaroerasmo.messaging.DetectionEventPublisher;
import com.icaroerasmo.pipeline.FrameContext;
import com.icaroerasmo.properties.DetectionProperties;
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

/**
 * Service for detecting people using the shared YOLOv8n model
 * ({@link YoloDetector}). COCO person class 0 detections above the configured
 * confidence are kept; false-positive person boxes are suppressed when they overlap
 * a car (COCO class 2) or a dog/cat (COCO class 16/15 - the model occasionally
 * misclassifies a dog as person with higher confidence than dog), or when their area
 * exceeds {@code maxPersonAreaRatio}.
 *
 * <p>The public {@link #detect} remains {@code synchronized} to guard the shared
 * model/net across cameras. Every DNN forward pass happens inside
 * {@link YoloDetector#detectRaw}, which runs under
 * {@link com.icaroerasmo.detectors.person.helper.DnnInferenceCoordinatorHelper#runExclusive}.
 * Returned {@link Rect}s are owned by the caller.
 */
@Log4j2
@Service
public class PersonDetector implements IDetector {

    private static final double CAR_SUPPRESSION_IOU = 0.35;
    private static final double PET_SUPPRESSION_IOU = 0.35;

    private final YoloDetector yoloDetector;
    private final double personConfidenceThreshold;
    private final double carConfidenceThreshold;
    private final double petConfidenceThreshold;
    private final double maxPersonAreaRatio;
    private final DetectionEventPublisher detectionEventPublisher;

    public PersonDetector(
            YoloDetector yoloDetector,
            ObjectDetectionProperties objectDetectionProperties,
            DetectionEventPublisher detectionEventPublisher
    ) {
        this.yoloDetector = yoloDetector;
        DetectionProperties detection = objectDetectionProperties.getDetection();
        this.personConfidenceThreshold = normalizeConfidenceThreshold(
                detection.getPersonConfidenceThreshold());
        this.carConfidenceThreshold = normalizeConfidenceThreshold(
                detection.getCarConfidenceThreshold());
        this.petConfidenceThreshold = normalizeConfidenceThreshold(
                detection.getPet().getConfidenceThreshold());
        this.maxPersonAreaRatio = Math.max(0.0, Math.min(1.0,
                detection.getMaxPersonAreaRatio()));
        this.detectionEventPublisher = detectionEventPublisher;
    }

    @Override
    public synchronized List<Rect> detect(FrameContext frameCtx) {
        Mat image = frameCtx.getFrame();
        if (image == null || image.empty()) {
            log.warn("Cannot detect people in null or empty image");
            return new ArrayList<>();
        }

        List<CocoDetection> detections = yoloDetector.detectRaw(image);
        if (detections.isEmpty()) {
            return new ArrayList<>();
        }

        int originalWidth = image.size().width();
        int originalHeight = image.size().height();

        List<Rect> candidates = new ArrayList<>();
        List<CocoDetection> cars = new ArrayList<>();
        List<CocoDetection> pets = new ArrayList<>();
        // Rects transferred to the returned list must NOT be released by the finally block.
        Set<Rect> returned = Collections.newSetFromMap(new IdentityHashMap<>());

        try {
            for (CocoDetection detection : detections) {
                DetectionClassFilter.DetectionType type = DetectionClassFilter.classify(
                        detection.classId(), detection.confidence(),
                        personConfidenceThreshold, carConfidenceThreshold, petConfidenceThreshold);
                switch (type) {
                    case PERSON -> candidates.add(detection.rect());
                    case CAR -> cars.add(detection);
                    case PET -> pets.add(detection);
                    default -> { /* other classes ignored */ }
                }
            }

            if (!candidates.isEmpty()) {
                detectionEventPublisher.publishPresence(frameCtx.getCameraName());
                log.debug("Detected {} person(s) in image", candidates.size());
            }

            if (!cars.isEmpty()) {
                detectionEventPublisher.publishCar(frameCtx.getCameraName());
                log.debug("Detected {} car(s) in image", cars.size());
            }

            // Suppress false-positive person boxes:
            // 1. Person box overlapping a detected car box with IoU > CAR_SUPPRESSION_IOU
            // 2. Person box overlapping a detected dog/cat box with IoU > PET_SUPPRESSION_IOU
            // 3. Person box whose area exceeds maxPersonAreaRatio of the frame area
            double frameArea = (double) originalWidth * originalHeight;
            List<Rect> survivors = new ArrayList<>(candidates.size());
            for (Rect person : candidates) {
                boolean suppressed = person.width() * person.height() > maxPersonAreaRatio * frameArea;
                if (!suppressed) {
                    for (CocoDetection car : cars) {
                        if (iou(person, car.rect()) > CAR_SUPPRESSION_IOU) {
                            suppressed = true;
                            break;
                        }
                    }
                }
                if (!suppressed) {
                    for (CocoDetection pet : pets) {
                        Rect petRect = pet.rect();
                        if (DetectionClassFilter.shouldSuppress(
                                DetectionClassFilter.intersectionOverUnion(
                                        person.x(), person.y(), person.width(), person.height(),
                                        petRect.x(), petRect.y(), petRect.width(), petRect.height()),
                                PET_SUPPRESSION_IOU)) {
                            suppressed = true;
                            break;
                        }
                    }
                }
                if (suppressed) {
                    log.debug("Suppressing person box at ({}, {}) size {}x{} (car/pet overlap or area exceeds {} of frame)",
                            person.x(), person.y(), person.width(), person.height(),
                            String.format("%.2f", maxPersonAreaRatio));
                } else {
                    survivors.add(person);
                    returned.add(person);
                }
            }
            return survivors;
        } catch (Exception e) {
            log.error("Error during person detection: {}", e.getMessage(), e);
            // No rects may leave this method; clear the returned-set so the finally
            // block deallocates every detection rect exactly once.
            returned.clear();
            return new ArrayList<>();
        } finally {
            // Release every rect that was not transferred to the caller (car, pet and
            // suppressed-person rects are all released here, exactly once).
            for (CocoDetection detection : detections) {
                Rect rect = detection.rect();
                if (rect != null && !returned.contains(rect)) {
                    rect.deallocate();
                }
            }
        }
    }

    private static double normalizeConfidenceThreshold(double threshold) {
        return Math.max(0.0, Math.min(1.0, threshold));
    }

    private static double iou(Rect p, Rect car) {
        int ix1 = Math.max(p.x(), car.x());
        int iy1 = Math.max(p.y(), car.y());
        int ix2 = Math.min(p.x() + p.width(), car.x() + car.width());
        int iy2 = Math.min(p.y() + p.height(), car.y() + car.height());
        if (ix2 <= ix1 || iy2 <= iy1) {
            return 0.0;
        }
        double inter = (double) (ix2 - ix1) * (iy2 - iy1);
        double union = (double) p.width() * p.height()
                + (double) car.width() * car.height() - inter;
        return union <= 0 ? 0.0 : inter / union;
    }
}
