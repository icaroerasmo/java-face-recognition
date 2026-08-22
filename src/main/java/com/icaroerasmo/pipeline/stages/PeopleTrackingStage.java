package com.icaroerasmo.pipeline.stages;

import com.icaroerasmo.detectors.person.services.PeopleTrackingService;
import com.icaroerasmo.messaging.DetectionEventPublisher;
import com.icaroerasmo.model.FaceRecognition;
import com.icaroerasmo.pipeline.FrameContext;
import com.icaroerasmo.pipeline.FrameStage;
import com.icaroerasmo.processing.FrameEncodingService;
import com.icaroerasmo.properties.ObjectDetectionProperties;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static com.icaroerasmo.processing.PersonAssociationGeometry.findIdentityForPerson;
import static com.icaroerasmo.processing.PersonAssociationGeometry.findPersonRectForFace;
import static com.icaroerasmo.utils.Constants.DESIRED_SCORE;
import static com.icaroerasmo.utils.FaceHashUtils.computePerceptualHash;

/**
 * STEP 3: Publishes the low-latency presence event, then decides between
 * unknown-people tracking (when no faces were recognized) and per-face
 * tracking with identities via {@link PeopleTrackingService}.
 *
 * <p>The rects it reads from the context are owned by the context and are NOT
 * deallocated here - {@link PeopleTrackingService} clones the rects it keeps.
 */
@Log4j2
@Component
@RequiredArgsConstructor
public class PeopleTrackingStage implements FrameStage {

    private final PeopleTrackingService peopleTrackingService;
    private final DetectionEventPublisher detectionEventPublisher;
    private final FrameEncodingService frameEncodingService;
    private final ObjectDetectionProperties objectDetectionProperties;

    @Override
    public void process(FrameContext ctx) {
        String cameraName = ctx.getCameraName();
        Mat img = ctx.getFrame();
        List<Rect> detectedPeople = ctx.getDetectedPeople();

        List<FaceRecognition.DetectedFaces> faces =
                ctx.getFaceRecognition() != null ? ctx.getFaceRecognition().getFaces() : null;

        // Publish low-latency presence event for the live-stream overlay (debounced)
        detectionEventPublisher.publishPresence(cameraName);

        // STEP 3: Check if faces were detected
        if (faces == null || faces.isEmpty()) {
            trackUnknownPeople(ctx);

            ctx.markProcessingComplete();
            return; // Don't continue processing if no faces detected
        }

        // Collect detected names with their distance scores
        Map<String, Double> detectedPeopleWithScores = faces.stream()
                .collect(Collectors.toMap(
                        FaceRecognition.DetectedFaces::getPersonName,
                        FaceRecognition.DetectedFaces::getDistance,
                        (existing, replacement) -> Math.min(existing, replacement) // Keep lowest distance if duplicate
                ));

        // Find the lowest distance score across all detections
        double lowestDistance = faces.stream()
                .mapToDouble(FaceRecognition.DetectedFaces::getDistance)
                .min()
                .orElse(DESIRED_SCORE);

        // FaceRecognitionService already applies its size-adaptive rejection threshold.
        String namesStr = String.join(", ", detectedPeopleWithScores.keySet());
        log.info("Pessoas detectadas em '{}': {} (lowest distance: {})",
                cameraName, namesStr, String.format("%.2f", lowestDistance));

        // Publish to Telegram with detected people information
        try {
            // Convert Mat to byte array WITHOUT drawing rectangles yet
            // Drawing will happen in PeopleTrackingService after person is tracked across frames
            byte[] imageBytes = frameEncodingService.encodeJpeg(img);

            // Keep annotations and tracking geometry based on full person detections.
            // Face rectangles vary sharply as people turn or approach the camera.
            List<PeopleTrackingService.PersonDetection> allPeopleDetections = detectedPeople.stream()
                    .map(personRect -> new PeopleTrackingService.PersonDetection(
                            findIdentityForPerson(personRect, faces),
                            personRect
                    ))
                    .collect(Collectors.toList());

            // Track each detected face using its enclosing person's geometry.
            List<FaceRecognition.DetectedFaces> orderedFaces = faces.stream()
                    .sorted(java.util.Comparator.comparingDouble(FaceRecognition.DetectedFaces::getDistance))
                    .toList();
            List<Rect> trackedPersonRects = new ArrayList<>();

            for (int faceIdx = 0; faceIdx < orderedFaces.size(); faceIdx++) {
                FaceRecognition.DetectedFaces face = orderedFaces.get(faceIdx);

                Rect trackingRect = findPersonRectForFace(face.getFaceRect(), detectedPeople);
                if (trackingRect == null) {
                    log.warn("⚠️ SKIPPED: no tracking rectangle for face #{} in camera '{}'", faceIdx + 1, cameraName);
                    continue;
                }
                if (trackedPersonRects.stream().anyMatch(rect -> rect == trackingRect)) {
                    log.debug("Skipping additional face associated with an already tracked person rectangle");
                    continue;
                }
                trackedPersonRects.add(trackingRect);

                // Hash the full person crop in both recognized and no-face frames so
                // the visual signature remains comparable across the whole track.
                byte[] faceHash = null;
                if (trackingRect != null) {
                    try {
                        byte[] encodedPerson = frameEncodingService.encodeRegionJpeg(img, trackingRect);
                        faceHash = computePerceptualHash(encodedPerson);
                    } catch (Exception e) {
                        log.warn("Failed to extract face region {} for camera '{}': {}",
                            faceIdx + 1, cameraName, e.getMessage());
                        continue; // Skip this face
                    }
                }

                log.info("🔍 FRAME PROCESSING: camera='{}', face #{}, person='{}', distance={}, rect={}, faceHash={}, imageBytes={}",
                    cameraName,
                    faceIdx + 1,
                    face.getPersonName(),
                    String.format("%.2f", face.getDistance()),
                    (trackingRect != null),
                    (faceHash != null ? faceHash.length : 0),
                    (imageBytes != null ? imageBytes.length : 0));

                if (faceHash == null) {
                    log.warn("⚠️ SKIPPED: faceHash is null for face #{} in camera '{}'", faceIdx + 1, cameraName);
                    continue;
                }

                log.info("📊 CALLING TRACKING SERVICE: camera='{}', face #{}, person='{}'",
                    cameraName, faceIdx + 1, face.getPersonName());

                // Track this face with all people detections for drawing
                PeopleTrackingService.TrackingResult trackingResult = peopleTrackingService.trackFace(
                    cameraName,
                    face.getPersonName(), // This person's name
                    trackingRect,         // Stable full-person rectangle
                    faceHash,             // Full-person visual hash
                    face.getDistance(),   // This person's distance
                    imageBytes,           // Full frame image bytes
                    allPeopleDetections,  // ALL detected people with names and rectangles for drawing
                    true
                );

                log.info("📋 TRACKING RESULT: camera='{}', face #{}, shouldSend={}, personName={}, score={}",
                    cameraName,
                    faceIdx + 1,
                    trackingResult.isShouldSend(),
                    trackingResult.getPersonName(),
                    trackingResult.isShouldSend() ? String.format("%.2f", trackingResult.getBestDistance()) : "N/A");

                // When this person's tracking is ready, notification will have ALL faces highlighted
                if (trackingResult.isShouldSend()) {
                    log.info("✅ VERDICT REACHED! Face #{} tracked successfully - notification sent with {} faces highlighted for '{}'",
                        faceIdx + 1, allPeopleDetections.size(), trackingResult.getPersonName());
                } else {
                    log.debug("⏳ STILL TRACKING: camera='{}', face #{} not ready to send yet",
                        cameraName, faceIdx + 1);
                }
            }

            if (faces.isEmpty()) {
                log.debug("📭 NO FACES: Skipping frame from camera '{}'", cameraName);
            }

        } catch (Exception e) {
            log.error("Failed to publish detection to Telegram for camera '{}': {}", cameraName, e.getMessage(), e);
        }
    }

    /**
     * People detected but no faces recognized - track ALL people individually
     * as unknown (previously the runner's {@code drawRectanglesOnPeople} logic).
     */
    private void trackUnknownPeople(FrameContext ctx) {
        String cameraName = ctx.getCameraName();
        Mat img = ctx.getFrame();
        List<Rect> detectedPeople = ctx.getDetectedPeople();

        if (objectDetectionProperties.getEnabled()) {
            // People detected but NO FACES - track ALL people individually
            log.info("Camera '{}': {} people detected but no faces recognized - tracking all people",
                    cameraName, detectedPeople.size());
        } else {
            log.info("Camera '{}': {} people detected (face recognition disabled) - tracking all people",
                    cameraName, detectedPeople.size());
        }

        try {
            // Convert full frame to byte array once (reused for all people)
            byte[] fullFrameBytes = null;
            try {
                fullFrameBytes = frameEncodingService.encodeJpeg(img);
            } catch (Exception e) {
                log.warn("Failed to convert frame to bytes: {}", e.getMessage());
            }

            if (fullFrameBytes != null) {
                // Convert rectangles to PersonDetection objects (all unknown at this point)
                List<PeopleTrackingService.PersonDetection> allPeopleDetections = detectedPeople.stream()
                    .map(rect -> new PeopleTrackingService.PersonDetection("Unknown", rect))
                    .collect(Collectors.toList());

                // Track EACH person individually
                for (int i = 0; i < detectedPeople.size(); i++) {
                    Rect personRect = detectedPeople.get(i);

                    // Extract person region for tracking
                    byte[] personHash = null;
                    try {
                        byte[] encodedPerson = frameEncodingService.encodeRegionJpeg(img, personRect);
                        personHash = computePerceptualHash(encodedPerson);
                    } catch (Exception e) {
                        log.warn("Failed to extract person region {}: {}", i + 1, e.getMessage());
                        continue; // Skip this person
                    }

                    // Track this person, passing ALL people detections for drawing
                    if (personHash != null) {
                        PeopleTrackingService.TrackingResult trackingResult = peopleTrackingService.trackFace(
                                cameraName,
                            "Unknown",
                            personRect,
                            personHash,
                            DESIRED_SCORE, // High distance that it's unknown
                            fullFrameBytes,
                            allPeopleDetections, // Pass ALL detected people with names for drawing
                            false
                        );

                        // When this person's tracking is ready, notification will have ALL people highlighted
                        if (trackingResult.isShouldSend()) {
                            log.info("Camera '{}': Person #{} tracked successfully - notification sent with {} people highlighted",
                                    cameraName, i + 1, detectedPeople.size());
                        } else {
                            log.debug("Camera '{}': Still tracking person #{} (total {} people)",
                                    cameraName, i + 1, detectedPeople.size());
                        }
                    }
                }
            }
        } catch (Exception e) {
            log.error("Failed to process unknown people for camera '{}': {}", cameraName, e.getMessage(), e);
        }
    }
}
