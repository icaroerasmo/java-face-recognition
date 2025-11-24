package com.icaroerasmo.service;

import com.icaroerasmo.utils.Constants;
import com.icaroerasmo.utils.MatUtil;
import lombok.Data;
import lombok.extern.log4j.Log4j2;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.*;
import jakarta.annotation.PreDestroy;

/**
 * Service to track people (both known and unknown) across multiple frames.
 * This helps prevent duplicate notifications for the same person and ensures
 * notifications are only sent for legitimate detections with movement.
 */
@Log4j2
@Service
public class PeopleTrackingService {

    private final TelegramPublisherService telegramPublisherService;
    private final DetectionHistoryService detectionHistoryService;
    private final MatUtil matUtil;
    private final ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(4);

    // Track people per camera
    private final Map<String, List<TrackedPerson>> trackedPeople = new ConcurrentHashMap<>();

    // Timeout for sending notification if person not detected anymore (5 seconds for better consolidation)
    private static final long NOTIFICATION_TIMEOUT_MS = 5000;

    public PeopleTrackingService(TelegramPublisherService telegramPublisherService,
                               DetectionHistoryService detectionHistoryService,
                               MatUtil matUtil) {
        this.telegramPublisherService = telegramPublisherService;
        this.detectionHistoryService = detectionHistoryService;
        this.matUtil = matUtil;
    }

    @PreDestroy
    public void shutdown() {
        log.info("Shutting down PeopleTrackingService scheduler");
        scheduler.shutdown();
        try {
            if (!scheduler.awaitTermination(5, TimeUnit.SECONDS)) {
                scheduler.shutdownNow();
            }
        } catch (InterruptedException e) {
            scheduler.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }

    // Minimum number of frames to track before sending notification
    // Requires more frames (5 * FPS = 150 frames at 30fps -> 5 seconds of frames) to avoid spurious detections
    private static final int MIN_TRACKING_FRAMES = (int) (Constants.FPS * 5);

    // Maximum time to track a person (15 seconds instead of 10)
    private static final long MAX_TRACKING_TIME_MS = 15 * 1000;

    // Maximum distance between face positions to consider it the same face (pixels)
    private static final double MAX_POSITION_DISTANCE = 300.0;

    // Face similarity threshold for tracking (extremely lenient)
    private static final int TRACKING_SIMILARITY_THRESHOLD = 150;

    // Minimum tracking duration before sending (2 seconds for better confidence)
    private static final long MIN_TRACKING_DURATION_MS = 2000;

    // Minimum movement required to consider person as real (increased to 50px)
    private static final double MIN_MOVEMENT_PIXELS = 50.0;

    // Maximum time gap between detections to keep tracking alive (20 seconds)
    private static final long TRACK_TIMEOUT_MS = 20000;

    /**
     * Result of tracking containing best frame information and determined identity
     */
    @Data
    public static class TrackingResult {
        private final boolean shouldSend;
        private final String personName; // Determined identity (most common across frames)
        private final byte[] bestFaceHash;
        private final double bestConfidenceScore;
        private final byte[] bestImageBytes; // Image bytes of the best frame

        public static TrackingResult notReady() {
            return new TrackingResult(false, null, null, 0.0, null);
        }
    }

    /**
     * Schedule a timeout notification that will be sent if person is not detected again.
     * This ensures notifications are sent even if person leaves before threshold is reached.
     * Now with longer timeout (5s) to better consolidate detections.
     */
    private void scheduleTimeoutNotification(TrackedPerson track, String cameraName, List<TrackedPerson> cameraTrackedPeople) {
        track.pendingNotification = scheduler.schedule(() -> {
            try {
                // Check if track still exists (might have been removed/sent already)
                if (!cameraTrackedPeople.contains(track)) {
                    log.debug("Track already removed, skipping timeout notification");
                    return;
                }

                int frameCount = track.observations.size();
                long trackingDuration = System.currentTimeMillis() - track.firstSeen;
                double distanceMoved = track.getTotalDistanceMoved();

                // Determine identity from accumulated observations
                String determinedIdentity = track.getMostCommonIdentity();

                log.info("⏰ TIMEOUT NOTIFICATION: Camera '{}' - Person '{}' disappeared after {} frames over {}ms (moved {}px) - sending notification",
                        cameraName, determinedIdentity, frameCount, trackingDuration, (int)distanceMoved);

                // Get best frame data
                byte[] bestFaceHash = track.getBestFaceHash();
                double bestScore = track.getBestConfidenceScore();
                byte[] bestImageBytes = track.getBestImageBytes();
                Rect bestRect = track.getBestRect();

                // Send notification
                sendNotificationNow(cameraName, determinedIdentity, bestScore, bestImageBytes, bestFaceHash, bestRect);

                // Remove from tracking and cleanup
                cameraTrackedPeople.remove(track);
                track.cleanup();

            } catch (Exception e) {
                log.error("Error in timeout notification for camera '{}': {}", cameraName, e.getMessage(), e);
            }
        }, NOTIFICATION_TIMEOUT_MS, TimeUnit.MILLISECONDS);

        log.debug("Scheduled timeout notification in {}ms for tracked person", NOTIFICATION_TIMEOUT_MS);
    }

    /**
     * Send notification immediately with cooldown check.
     * Draws rectangle around the detected person/face on the image before sending.
     */
    private void sendNotificationNow(String cameraName, String determinedIdentity, double bestScore,
                                     byte[] bestImageBytes, byte[] bestFaceHash, Rect personRect) {
        Mat annotatedImg = null;
        try {
            log.info("🔔 SENDING NOTIFICATION: camera='{}', identity='{}', score={}",
                cameraName, determinedIdentity, String.format("%.2f", bestScore));

            if (bestImageBytes == null || bestImageBytes.length == 0) {
                log.error("Cannot send notification: image bytes is null or empty");
                return;
            }

            boolean isUnknown = "Unknown".equalsIgnoreCase(determinedIdentity);

            // Convert image bytes to Mat so we can draw on it
            org.bytedeco.javacpp.BytePointer imagePointer = new org.bytedeco.javacpp.BytePointer(bestImageBytes);
            Mat originalImg = org.bytedeco.opencv.global.opencv_imgcodecs.imdecode(
                new org.bytedeco.opencv.opencv_core.Mat(imagePointer),
                org.bytedeco.opencv.global.opencv_imgcodecs.IMREAD_COLOR
            );
            imagePointer.deallocate();

            if (originalImg.empty()) {
                log.error("Failed to decode image bytes to Mat");
                return;
            }

            // Clone the image and draw rectangle on it
            annotatedImg = originalImg.clone();
            matUtil.releaseResources(originalImg); // Release original, keep annotated

            if (personRect != null) {
                // Normalize "Unknown Person" to just "Unknown"
                String displayName = determinedIdentity;
                if (displayName != null && displayName.toLowerCase().contains("unknown")) {
                    displayName = "Unknown";
                }

                // Draw rectangle and name on the annotated image
                matUtil.drawRectangleAndName(annotatedImg, displayName, personRect);
                log.debug("Drew rectangle for '{}' at ({}, {}, {}, {})",
                    displayName, personRect.x(), personRect.y(), personRect.width(), personRect.height());
            }

            // Convert annotated image back to bytes
            org.bytedeco.javacpp.BytePointer buf = new org.bytedeco.javacpp.BytePointer();
            org.bytedeco.javacpp.BytePointer jpgExt = new org.bytedeco.javacpp.BytePointer(".jpg");
            org.bytedeco.opencv.global.opencv_imgcodecs.imencode(jpgExt, annotatedImg, buf);
            byte[] annotatedImageBytes = new byte[(int) buf.limit()];
            buf.get(annotatedImageBytes);
            buf.deallocate();
            jpgExt.deallocate();

            // Compute image hash for detection history
            String imageHash = detectionHistoryService.computeImageHash(annotatedImageBytes);

            // Check if we recently sent this person (using appropriate method)
            boolean shouldSend = isUnknown
                    ? detectionHistoryService.shouldSendUnknownDetection(imageHash, determinedIdentity, cameraName, bestFaceHash)
                    : detectionHistoryService.shouldSendDetection(imageHash, determinedIdentity, cameraName, bestFaceHash);

            log.info("📊 COOLDOWN CHECK: shouldSend={}, isUnknown={}, identity={}",
                shouldSend, isUnknown, determinedIdentity);

            if (shouldSend) {
                // Create scores map with determined identity and best score
                Map<String, Double> bestScores = Map.of(determinedIdentity, bestScore);

                log.info("📤 CALLING TELEGRAM API: imageSize={} bytes", annotatedImageBytes.length);

                // Send notification with annotated image (with rectangle) and determined identity
                telegramPublisherService.publishDetection(annotatedImageBytes, bestScores, cameraName);

                log.info("✅ NOTIFICATION SENT SUCCESSFULLY for '{}'", determinedIdentity);

                // Mark as sent to prevent duplicates
                if (isUnknown) {
                    detectionHistoryService.markUnknownDetectionAsSent(imageHash, determinedIdentity, cameraName, bestFaceHash);
                }
            } else {
                log.info("⏭️ SKIPPING: '{}' already sent recently for camera '{}'", determinedIdentity, cameraName);
            }
        } catch (Exception e) {
            log.error("❌ FAILED to send notification for camera '{}': {}", cameraName, e.getMessage(), e);
        } finally {
            // Clean up the annotated image
            if (annotatedImg != null) {
                matUtil.releaseResources(annotatedImg);
            }
        }
    }

    /**
     * Track a person across multiple frames to determine if notification should be sent.
     * Returns TrackingResult with most common identity and best frame if ready to send notification.
     * Improved to prevent duplicate notifications and require more confident detections.
     *
     * @param cameraName Name of the camera
     * @param personName Detected person name (or "Unknown")
     * @param faceRect Rectangle of the detected face/person
     * @param faceHash Hash of the face image for similarity comparison
     * @param confidenceScore Recognition confidence score (lower is better)
     * @param imageBytes Image bytes of the current frame
     * @return TrackingResult indicating if should send, determined identity, and best frame data
     */
    public TrackingResult trackFace(String cameraName, String personName, Rect faceRect, byte[] faceHash, double confidenceScore, byte[] imageBytes) {
        if (faceRect == null || faceHash == null || imageBytes == null) {
            return TrackingResult.notReady(); // Can't track without face data
        }

        long now = System.currentTimeMillis();
        List<TrackedPerson> cameraTrackedPeople = trackedPeople.computeIfAbsent(cameraName, k -> new ArrayList<>());

        // Find if this person matches an existing tracked person
        TrackedPerson matchedTrack = findMatchingTrackedPerson(cameraTrackedPeople, faceRect, faceHash, now);

        if (matchedTrack != null) {
            // Update existing track - clone Rect to avoid issues if original is deallocated
            Rect clonedRect = new Rect(faceRect.x(), faceRect.y(), faceRect.width(), faceRect.height());
            matchedTrack.addObservation(clonedRect, faceHash, now, confidenceScore, personName, imageBytes);

            long trackingDuration = now - matchedTrack.firstSeen;
            int frameCount = matchedTrack.observations.size();

            log.debug("Camera '{}': Tracking person - {} frames over {}ms (distance moved: {}px, best score: {}, current: {})",
                    cameraName, frameCount, trackingDuration, (int)matchedTrack.getTotalDistanceMoved(),
                    String.format("%.2f", matchedTrack.getBestConfidenceScore()), personName);

            // Cancel any existing timeout notification and schedule a new one
            // This ensures notification is sent if person disappears before reaching threshold
            matchedTrack.cancelPendingNotification();
            scheduleTimeoutNotification(matchedTrack, cameraName, cameraTrackedPeople);

            // Check if we've tracked long enough
            if (frameCount >= MIN_TRACKING_FRAMES && trackingDuration >= MIN_TRACKING_DURATION_MS) {
                // Check if person is actually moving (not static)
                double distanceMoved = matchedTrack.getTotalDistanceMoved();
                if (distanceMoved > MIN_MOVEMENT_PIXELS) {
                    // Determine the most common identity
                    String determinedIdentity = matchedTrack.getMostCommonIdentity();

                    log.info("Camera '{}': Person tracked through {} frames over {}ms, moved {}px, determined identity: '{}', best score: {} - ready to send notification",
                            cameraName, frameCount, trackingDuration, (int)distanceMoved, determinedIdentity,
                            String.format("%.2f", matchedTrack.getBestConfidenceScore()));

                    // Cancel timeout since we're sending immediately
                    matchedTrack.cancelPendingNotification();

                    // Get best frame data before cleanup
                    byte[] bestFaceHash = matchedTrack.getBestFaceHash();
                    double bestScore = matchedTrack.getBestConfidenceScore();
                    byte[] bestImageBytes = matchedTrack.getBestImageBytes();
                    Rect bestRect = matchedTrack.getBestRect();

                    // Send notification immediately
                    sendNotificationNow(cameraName, determinedIdentity, bestScore, bestImageBytes, bestFaceHash, bestRect);

                    // Remove from tracking and cleanup resources
                    cameraTrackedPeople.remove(matchedTrack);
                    matchedTrack.cleanup();
                    return new TrackingResult(true, determinedIdentity, bestFaceHash, bestScore, bestImageBytes);
                } else {
                    log.debug("Camera '{}': Person appears static (moved only {}px) - continuing to track",
                            cameraName, (int)distanceMoved);
                }
            }

            // Check if tracking expired (tracked too long without movement)
            if (trackingDuration > MAX_TRACKING_TIME_MS) {
                double distanceMoved = matchedTrack.getTotalDistanceMoved();
                if (distanceMoved < MIN_MOVEMENT_PIXELS) {
                    log.warn("Camera '{}': Person tracked for {}ms but barely moved ({}px) - likely detection artifact, discarding",
                            cameraName, trackingDuration, (int)distanceMoved);
                    cameraTrackedPeople.remove(matchedTrack);
                    matchedTrack.cleanup();
                    return TrackingResult.notReady(); // Don't send - likely false positive
                } else {
                    // Determine the most common identity
                    String determinedIdentity = matchedTrack.getMostCommonIdentity();

                    log.info("Camera '{}': Person tracking expired after {}ms, moved {}px, determined identity: '{}', best score: {} - sending notification",
                            cameraName, trackingDuration, (int)distanceMoved, determinedIdentity,
                            String.format("%.2f", matchedTrack.getBestConfidenceScore()));

                    // Get best frame data before cleanup
                    byte[] bestFaceHash = matchedTrack.getBestFaceHash();
                    double bestScore = matchedTrack.getBestConfidenceScore();
                    byte[] bestImageBytes = matchedTrack.getBestImageBytes();
                    Rect bestRect = matchedTrack.getBestRect();

                    // Send notification
                    sendNotificationNow(cameraName, determinedIdentity, bestScore, bestImageBytes, bestFaceHash, bestRect);

                    cameraTrackedPeople.remove(matchedTrack);
                    matchedTrack.cleanup();
                    return new TrackingResult(true, determinedIdentity, bestFaceHash, bestScore, bestImageBytes);
                }
            }

            return TrackingResult.notReady(); // Still tracking
        } else {
            // New person, start tracking - clone Rect to avoid issues if original is deallocated
            Rect clonedRect = new Rect(faceRect.x(), faceRect.y(), faceRect.width(), faceRect.height());
            TrackedPerson newTrack = new TrackedPerson(clonedRect, faceHash, now, confidenceScore, personName, imageBytes);
            cameraTrackedPeople.add(newTrack);

            // Schedule timeout notification in case person disappears
            scheduleTimeoutNotification(newTrack, cameraName, cameraTrackedPeople);

            log.debug("Camera '{}': Started tracking new person '{}' at position ({}, {}) with score {} (timeout notification scheduled)",
                    cameraName, personName, faceRect.x(), faceRect.y(), String.format("%.2f", confidenceScore));
            return TrackingResult.notReady(); // Just started tracking
        }
    }


    /**
     * Find a tracked person that matches the current detection
     */
    private TrackedPerson findMatchingTrackedPerson(List<TrackedPerson> tracks, Rect faceRect, byte[] faceHash, long currentTime) {
        log.debug("🔍 MATCHING: Checking {} existing tracks for person at ({}, {})", tracks.size(), faceRect.x(), faceRect.y());

        for (TrackedPerson track : tracks) {
            Rect lastRect = track.getLatestRect();
            long timeSinceLastSeen = currentTime - track.lastSeen;

            // Check if track is still recent (allow 5 second gaps instead of 2)
            if (timeSinceLastSeen > TRACK_TIMEOUT_MS) {
                log.debug("❌ Track at ({}, {}) is too old: {}ms > {}ms",
                    lastRect.x(), lastRect.y(), timeSinceLastSeen, TRACK_TIMEOUT_MS);
                continue; // Track is too old
            }

            // Check position distance
            double distance = calculateDistance(faceRect, lastRect);
            if (distance > MAX_POSITION_DISTANCE) {
                log.debug("❌ Track at ({}, {}) is too far: {}px > {}px",
                    lastRect.x(), lastRect.y(), (int)distance, (int)MAX_POSITION_DISTANCE);
                continue; // Face moved too far
            }

            // Check face hash similarity (reactivated with high threshold)
            int similarity = computeFaceHashSimilarity(faceHash, track.getLatestFaceHash());
            log.debug("📊 Track at ({}, {}) similarity: {} (threshold: {}), distance: {}px, age: {}ms",
                lastRect.x(), lastRect.y(), similarity, TRACKING_SIMILARITY_THRESHOLD,
                (int)distance, timeSinceLastSeen);

            if (similarity <= TRACKING_SIMILARITY_THRESHOLD) {
                log.info("✅ MATCHED: Face at ({}, {}) matched to existing track (similarity: {}, distance: {}px)",
                    faceRect.x(), faceRect.y(), similarity, (int)distance);
                return track;
            } else {
                log.debug("❌ Similarity too high: {} > {} - faces too different", similarity, TRACKING_SIMILARITY_THRESHOLD);
            }
        }

        log.debug("🆕 NO MATCH: Creating new track for face at ({}, {})", faceRect.x(), faceRect.y());
        return null;
    }

    /**
     * Calculate distance between two face rectangles (center to center)
     */
    private double calculateDistance(Rect rect1, Rect rect2) {
        double centerX1 = rect1.x() + rect1.width() / 2.0;
        double centerY1 = rect1.y() + rect1.height() / 2.0;
        double centerX2 = rect2.x() + rect2.width() / 2.0;
        double centerY2 = rect2.y() + rect2.height() / 2.0;

        double dx = centerX1 - centerX2;
        double dy = centerY1 - centerY2;
        return Math.sqrt(dx * dx + dy * dy);
    }

    /**
     * Compute similarity between two face hashes
     */
    private int computeFaceHashSimilarity(byte[] hash1, byte[] hash2) {
        if (hash1 == null || hash2 == null) {
            return 100;
        }

        if (Math.abs(hash1.length - hash2.length) > hash1.length * 0.1) {
            return 100;
        }

        int minLength = Math.min(hash1.length, hash2.length);
        int differentBytes = 0;

        for (int i = 0; i < minLength; i++) {
            if (hash1[i] != hash2[i]) {
                differentBytes++;
            }
        }

        differentBytes += Math.abs(hash1.length - hash2.length);
        return Math.min(100, (differentBytes * 100) / Math.max(hash1.length, hash2.length));
    }

    /**
     * Cleanup old tracked people
     */
    public void cleanupOldTracks() {
        long cutoffTime = System.currentTimeMillis() - MAX_TRACKING_TIME_MS;
        trackedPeople.values().forEach(tracks -> {
            tracks.removeIf(track -> {
                if (track.lastSeen < cutoffTime) {
                    track.cleanup(); // Clean up native resources
                    return true;
                }
                return false;
            });
        });
        log.debug("Cleaned up old people tracks");
    }

    /**
     * Represents a person being tracked across multiple frames
     */
    @Data
    private static class TrackedPerson {
        private final long firstSeen;
        private long lastSeen;
        private final List<FaceObservation> observations = new ArrayList<>();
        private FaceObservation bestObservation; // Track the best scoring observation
        private ScheduledFuture<?> pendingNotification; // Timeout notification future

        public TrackedPerson(Rect initialRect, byte[] initialFaceHash, long timestamp, double confidenceScore, String personName, byte[] imageBytes) {
            this.firstSeen = timestamp;
            this.lastSeen = timestamp;
            FaceObservation observation = new FaceObservation(initialRect, initialFaceHash, timestamp, confidenceScore, personName, imageBytes);
            this.observations.add(observation);
            this.bestObservation = observation; // First is best by default
        }

        public void addObservation(Rect rect, byte[] faceHash, long timestamp, double confidenceScore, String personName, byte[] imageBytes) {
            this.lastSeen = timestamp;
            FaceObservation observation = new FaceObservation(rect, faceHash, timestamp, confidenceScore, personName, imageBytes);
            this.observations.add(observation);

            // Update best observation if this one has lower confidence (better match)
            if (bestObservation == null || confidenceScore < bestObservation.confidenceScore) {
                bestObservation = observation;
            }
        }

        public Rect getLatestRect() {
            return observations.get(observations.size() - 1).rect;
        }

        public byte[] getLatestFaceHash() {
            return observations.get(observations.size() - 1).faceHash;
        }


        public byte[] getBestFaceHash() {
            return bestObservation != null ? bestObservation.faceHash : getLatestFaceHash();
        }

        public Rect getBestRect() {
            return bestObservation != null ? bestObservation.rect : getLatestRect();
        }

        public double getBestConfidenceScore() {
            return bestObservation != null ? bestObservation.confidenceScore : 100.0;
        }

        public byte[] getBestImageBytes() {
            return bestObservation != null ? bestObservation.imageBytes : null;
        }

        /**
         * Determine the most common identity across all observations
         * @return The person name that appears most frequently, or "Unknown" if there's a tie
         */
        public String getMostCommonIdentity() {
            Map<String, Integer> identityCounts = new java.util.HashMap<>();

            // Count occurrences of each identity
            for (FaceObservation obs : observations) {
                String name = obs.personName != null ? obs.personName : "Unknown";
                identityCounts.put(name, identityCounts.getOrDefault(name, 0) + 1);
            }

            // Find the most common identity and check for ties
            String mostCommon = "Unknown";
            int maxCount = 0;
            int tieCount = 0; // Count how many identities share the max count

            for (Map.Entry<String, Integer> entry : identityCounts.entrySet()) {
                if (entry.getValue() > maxCount) {
                    maxCount = entry.getValue();
                    mostCommon = entry.getKey();
                    tieCount = 1; // Reset tie count
                } else if (entry.getValue() == maxCount) {
                    tieCount++; // Another identity with same count
                }
            }

            // If there's a tie between different identities, return Unknown
            if (tieCount > 1) {
                log.debug("Identity uncertain: tie detected with {} identities at {} occurrences - reporting as Unknown",
                    tieCount, maxCount);
                return "Unknown";
            }

            log.debug("Determined identity: '{}' appeared {} times out of {} observations",
                mostCommon, maxCount, observations.size());

            return mostCommon;
        }

        /**
         * Calculate total distance the face has moved
         */
        public double getTotalDistanceMoved() {
            if (observations.size() < 2) {
                return 0.0;
            }

            double totalDistance = 0.0;
            for (int i = 1; i < observations.size(); i++) {
                Rect prev = observations.get(i - 1).rect;
                Rect curr = observations.get(i).rect;

                double prevCenterX = prev.x() + prev.width() / 2.0;
                double prevCenterY = prev.y() + prev.height() / 2.0;
                double currCenterX = curr.x() + curr.width() / 2.0;
                double currCenterY = curr.y() + curr.height() / 2.0;

                double dx = currCenterX - prevCenterX;
                double dy = currCenterY - prevCenterY;
                totalDistance += Math.sqrt(dx * dx + dy * dy);
            }

            return totalDistance;
        }

        /**
         * Cancel any pending timeout notification
         */
        public void cancelPendingNotification() {
            if (pendingNotification != null && !pendingNotification.isDone()) {
                pendingNotification.cancel(false);
                log.debug("Cancelled pending timeout notification for tracked person");
            }
            pendingNotification = null;
        }

        /**
         * Clean up native resources (deallocate Rect objects) and cancel pending notifications
         */
        public void cleanup() {
            cancelPendingNotification();
            for (FaceObservation observation : observations) {
                if (observation.rect != null) {
                    observation.rect.deallocate();
                }
            }
            observations.clear();
        }
    }

    /**
     * Represents a single observation of a face
     */
    @Data
    private static class FaceObservation {
        private final Rect rect;
        private final byte[] faceHash;
        private final long timestamp;
        private final double confidenceScore; // Lower is better
        private final String personName; // Detected identity for this frame
        private final byte[] imageBytes; // Frame image for this observation
    }
}
