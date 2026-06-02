package com.icaroerasmo.service;

import com.icaroerasmo.utils.Constants;
import com.icaroerasmo.utils.MatUtil;
import lombok.AllArgsConstructor;
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
    private final GifCreationService gifCreationService;
    private final ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(4);

    // Track people per camera
    private final Map<String, List<TrackedPerson>> trackedPeople = new ConcurrentHashMap<>();

    // Timeout for sending notification if person not detected anymore (5 seconds for better consolidation)
    private static final long NOTIFICATION_TIMEOUT_MS = 5000;

    public PeopleTrackingService(TelegramPublisherService telegramPublisherService,
                               DetectionHistoryService detectionHistoryService,
                               MatUtil matUtil,
                               GifCreationService gifCreationService) {
        this.telegramPublisherService = telegramPublisherService;
        this.detectionHistoryService = detectionHistoryService;
        this.matUtil = matUtil;
        this.gifCreationService = gifCreationService;
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
    // 5 seconds of frames (150 frames at 30fps) to better handle people turning away
    private static final int MIN_TRACKING_FRAMES = (int) (Constants.FPS * 5);

    // Maximum time to track a person (20 seconds to handle people turning/walking away)
    private static final long MAX_TRACKING_TIME_MS = 20 * 1000;

    // Maximum distance between positions to consider it the same person (pixels)
    // Increased to 500px to track people walking away or across the frame
    private static final double MAX_POSITION_DISTANCE = 500.0;

    // Face similarity threshold for tracking (very lenient to handle face angle changes)
    // Set to 200 to allow significant variations when person turns their head/back
    private static final int TRACKING_SIMILARITY_THRESHOLD = 200;

    // Minimum tracking duration before sending (reduced to 1.5 seconds)
    private static final long MIN_TRACKING_DURATION_MS = 1500;

    // Minimum movement required to consider person as real (reduced to 30px to be more sensitive)
    private static final double MIN_MOVEMENT_PIXELS = 30.0;

    // Maximum time gap between detections to keep tracking alive (10 seconds)
    // Allows person to turn around, walk briefly, etc. without losing track
    private static final long TRACK_TIMEOUT_MS = 10000;

    /**
     * Result of tracking containing best frame information and determined identity
     */
    @Data
    public static class TrackingResult {
        private final boolean shouldSend;
        private final String personName; // Determined identity (most common across frames)
        private final byte[] bestFaceHash;
        private final double bestDistance;
        private final byte[] bestImageBytes; // Image bytes of the best frame
        private final List<PersonDetection> allPeople; // All detected people with names and rectangles
        private final int frameCount; // Number of frames person was tracked across

        public static TrackingResult notReady() {
            return new TrackingResult(false, null, null, 0.0, null, null, 0);
        }

        public static TrackingResult ready() {
            return new TrackingResult(true, "Unknown", null, 0.0, null, null, 0);
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
                IdentityResult identityResult = track.getMostCommonIdentity();
                String determinedIdentity = identityResult.getPersonName();
                int identityFrameCount = identityResult.getFrameCount();

                log.info("⏰ TIMEOUT NOTIFICATION: Camera '{}' - Person '{}' disappeared after {} frames over {}ms (moved {}px) - sending notification",
                        cameraName, determinedIdentity, frameCount, trackingDuration, (int)distanceMoved);

                // Get best frame data
                byte[] bestFaceHash = track.getBestFaceHash();
                double bestScore = track.getBestDistance();
                byte[] bestImageBytes = track.getBestImageBytes();
                List<PersonDetection> bestAllPeople = track.getBestAllPeople();
                List<byte[]> allFrameImages = track.getAllFrameImages();

                // Send notification with identity frame count and total tracked frames
                sendNotificationNow(cameraName, determinedIdentity, bestScore, bestImageBytes, bestFaceHash, bestAllPeople, identityFrameCount, frameCount, allFrameImages);

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
     * Draws rectangles around ALL detected people in the image with their corresponding names before sending.
     * After sending the notification, creates and sends a GIF of all tracked frames.
     */
    private void sendNotificationNow(String cameraName, String determinedIdentity, double bestScore,
                                     byte[] bestImageBytes, byte[] bestFaceHash, List<PersonDetection> allPeople,
                                     int identityFrameCount, int totalTrackedFrames, List<byte[]> allFrameImages) {
        Mat annotatedImg = null;
        try {
            log.info("🔔 SENDING NOTIFICATION: camera='{}', identity='{}', score={}, people count={}, identityFrames={}, totalFrames={}, totalFrameImages={}",
                cameraName, determinedIdentity, String.format("%.2f", bestScore),
                allPeople != null ? allPeople.size() : 0, identityFrameCount, totalTrackedFrames,
                allFrameImages != null ? allFrameImages.size() : 0);

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

            // Draw rectangles around ALL detected people using the DETERMINED identity from tracking
            // Since tracking happens per person, we only draw for the person being tracked
            if (allPeople != null && !allPeople.isEmpty()) {
                // Use the determined identity (the result of tracking, not frame detection)
                String displayName = determinedIdentity;

                for (PersonDetection person : allPeople) {
                    if (person != null && person.getRect() != null) {
                        // Draw rectangle with the FINAL determined identity (after tracking)
                        matUtil.drawRectangleAndName(annotatedImg, displayName, person.getRect());
                        log.debug("Drew rectangle for DETERMINED identity '{}' at ({}, {}, {}, {})",
                            displayName, person.getRect().x(), person.getRect().y(),
                            person.getRect().width(), person.getRect().height());
                    }
                }
                log.info("Drew {} rectangle(s) with determined identity: '{}'", allPeople.size(), displayName);
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

                log.info("📤 CALLING TELEGRAM API: imageSize={} bytes, identityFrameCount={}, totalTrackedFrames={}",
                    annotatedImageBytes.length, identityFrameCount, totalTrackedFrames);

                // Send notification with annotated image (with rectangle) and determined identity
                telegramPublisherService.publishDetection(annotatedImageBytes, bestScores, cameraName, identityFrameCount, totalTrackedFrames);

                log.info("✅ NOTIFICATION SENT SUCCESSFULLY for '{}' (identified in {} frames, tracked across {} total frames)",
                    determinedIdentity, identityFrameCount, totalTrackedFrames);

                // Mark as sent to prevent duplicates
                if (isUnknown) {
                    detectionHistoryService.markUnknownDetectionAsSent(imageHash, determinedIdentity, cameraName, bestFaceHash);
                }

                // Create and send GIF animation after the main notification
                // Only create GIF if we have at least 10 frames for meaningful animation
                if (allFrameImages != null && allFrameImages.size() >= 10) {
                    log.info("📹 Creating GIF from {} tracked frames for '{}'", allFrameImages.size(), determinedIdentity);

                    // Create GIF in a separate thread to not block
                    new Thread(() -> {
                        try {
                            byte[] gifBytes = gifCreationService.createGif(allFrameImages);
                            if (gifBytes != null && gifBytes.length > 0) {
                                String gifCaption = String.format(
                                    "<b>Tracking Animation</b>\n" +
                                    "<b>Camera:</b> %s\n" +
                                    "<b>Person:</b> %s\n" +
                                    "<b>Frames:</b> %d\n" +
                                    "<b>Duration:</b> ~%.1f seconds",
                                    cameraName, determinedIdentity, allFrameImages.size(),
                                    allFrameImages.size() / (double) gifCreationService.getGifFps()
                                );
                                telegramPublisherService.sendAnimation(gifBytes, gifCaption, cameraName);
                            } else {
                                log.warn("Failed to create GIF for '{}' - no bytes generated", determinedIdentity);
                            }
                        } catch (Exception e) {
                            log.error("Error creating/sending GIF for '{}': {}", determinedIdentity, e.getMessage(), e);
                        }
                    }, "GIF-Creator-" + cameraName).start();
                } else {
                    log.debug("Skipping GIF creation - not enough frames (have: {})",
                        allFrameImages != null ? allFrameImages.size() : 0);
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
     * @param distance Recognition distance score (lower is better)
     * @param imageBytes Image bytes of the current frame
     * @param allPeople All detected people with their names and rectangles
     * @return TrackingResult indicating if should send, determined identity, and best frame data
     */
    public TrackingResult trackFace(String cameraName, String personName, Rect faceRect, byte[] faceHash, double distance, byte[] imageBytes, List<PersonDetection> allPeople) {
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
            matchedTrack.addObservation(clonedRect, faceHash, now, distance, personName, imageBytes, allPeople);

            long trackingDuration = now - matchedTrack.firstSeen;
            int frameCount = matchedTrack.observations.size();

            log.debug("Camera '{}': Tracking person - {} frames over {}ms (distance moved: {}px, best score: {}, current: {})",
                    cameraName, frameCount, trackingDuration, (int)matchedTrack.getTotalDistanceMoved(),
                    String.format("%.2f", matchedTrack.getBestDistance()), personName);

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
                    IdentityResult identityResult = matchedTrack.getMostCommonIdentity();
                    String determinedIdentity = identityResult.getPersonName();
                    int identityFrameCount = identityResult.getFrameCount();

                    log.info("Camera '{}': Person tracked through {} frames over {}ms, moved {}px, determined identity: '{}', best score: {} - ready to send notification",
                            cameraName, frameCount, trackingDuration, (int)distanceMoved, determinedIdentity,
                            String.format("%.2f", matchedTrack.getBestDistance()));

                    // Cancel timeout since we're sending immediately
                    matchedTrack.cancelPendingNotification();

                    // Get best frame data before cleanup
                    byte[] bestFaceHash = matchedTrack.getBestFaceHash();
                    double bestScore = matchedTrack.getBestDistance();
                    byte[] bestImageBytes = matchedTrack.getBestImageBytes();
                    List<PersonDetection> bestAllPeople = matchedTrack.getBestAllPeople();
                    List<byte[]> allFrameImages = matchedTrack.getAllFrameImages();

                    // Send notification immediately with identity frame count and total tracked frames
                    sendNotificationNow(cameraName, determinedIdentity, bestScore, bestImageBytes, bestFaceHash, bestAllPeople, identityFrameCount, frameCount, allFrameImages);

                    // Remove from tracking and cleanup resources
                    cameraTrackedPeople.remove(matchedTrack);
                    matchedTrack.cleanup();
                    return new TrackingResult(true, determinedIdentity, bestFaceHash, bestScore, bestImageBytes, bestAllPeople, identityFrameCount);
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
                    IdentityResult identityResult = matchedTrack.getMostCommonIdentity();
                    String determinedIdentity = identityResult.getPersonName();
                    int identityFrameCount = identityResult.getFrameCount();

                    log.info("Camera '{}': Person tracking expired after {}ms, moved {}px, determined identity: '{}', best score: {} - sending notification",
                            cameraName, trackingDuration, (int)distanceMoved, determinedIdentity,
                            String.format("%.2f", matchedTrack.getBestDistance()));

                    // Get best frame data before cleanup
                    byte[] bestFaceHash = matchedTrack.getBestFaceHash();
                    double bestScore = matchedTrack.getBestDistance();
                    byte[] bestImageBytes = matchedTrack.getBestImageBytes();
                    List<PersonDetection> bestAllPeople = matchedTrack.getBestAllPeople();
                    List<byte[]> allFrameImages = matchedTrack.getAllFrameImages();

                    // Send notification with identity frame count and total tracked frames
                    sendNotificationNow(cameraName, determinedIdentity, bestScore, bestImageBytes, bestFaceHash, bestAllPeople, identityFrameCount, frameCount, allFrameImages);

                    cameraTrackedPeople.remove(matchedTrack);
                    matchedTrack.cleanup();
                    return new TrackingResult(true, determinedIdentity, bestFaceHash, bestScore, bestImageBytes, bestAllPeople, identityFrameCount);
                }
            }

            return TrackingResult.notReady(); // Still tracking
        } else {
            // New person, start tracking - clone Rect to avoid issues if original is deallocated
            Rect clonedRect = new Rect(faceRect.x(), faceRect.y(), faceRect.width(), faceRect.height());
            TrackedPerson newTrack = new TrackedPerson(
                clonedRect,
                faceHash,
                now,
                distance,
                personName,
                imageBytes,
                allPeople,
                gifCreationService.getMaxGifFrames()
            );
            cameraTrackedPeople.add(newTrack);

            // Schedule timeout notification in case person disappears
            scheduleTimeoutNotification(newTrack, cameraName, cameraTrackedPeople);

            log.debug("Camera '{}': Started tracking new person '{}' at position ({}, {}) with score {} (timeout notification scheduled)",
                    cameraName, personName, faceRect.x(), faceRect.y(), String.format("%.2f", distance));
            return TrackingResult.notReady(); // Just started tracking
        }
    }


    /**
     * Find a tracked person that matches the current detection.
     * Uses adaptive matching - more lenient for established tracks to handle people turning around.
     */
    private TrackedPerson findMatchingTrackedPerson(List<TrackedPerson> tracks, Rect faceRect, byte[] faceHash, long currentTime) {
        log.debug("🔍 MATCHING: Checking {} existing tracks for person at ({}, {})", tracks.size(), faceRect.x(), faceRect.y());

        for (TrackedPerson track : tracks) {
            Rect lastRect = track.getLatestRect();
            long timeSinceLastSeen = currentTime - track.lastSeen;
            long totalTrackingTime = currentTime - track.firstSeen;

            // Check if track exceeded maximum tracking time - stop tracking if so
            if (totalTrackingTime > MAX_TRACKING_TIME_MS) {
                log.debug("❌ Track at ({}, {}) exceeded max tracking time: {}ms > {}ms",
                    lastRect.x(), lastRect.y(), totalTrackingTime, MAX_TRACKING_TIME_MS);
                continue; // Track exceeded time limit, don't match
            }

            // Check if track is still recent
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
                continue; // Person moved too far
            }

            // Adaptive similarity threshold based on tracking duration
            // For established tracks (>2 seconds), be more lenient about face similarity
            // This helps track people who turn their back or change angles
            long trackDuration = currentTime - track.firstSeen;
            int adaptiveSimilarityThreshold = TRACKING_SIMILARITY_THRESHOLD;

            if (trackDuration > 2000) { // More than 2 seconds of tracking
                // For close distances (<150px), be very lenient - probably same person
                if (distance < 150) {
                    adaptiveSimilarityThreshold = 300; // Very lenient
                } else {
                    adaptiveSimilarityThreshold = 250; // Lenient
                }
                log.debug("📈 Using adaptive threshold {} (track age: {}ms, distance: {}px)",
                    adaptiveSimilarityThreshold, trackDuration, (int)distance);
            }

            // Check face hash similarity
            int similarity = computeFaceHashSimilarity(faceHash, track.getLatestFaceHash());
            log.debug("📊 Track at ({}, {}) similarity: {} (threshold: {}), distance: {}px, age: {}ms",
                lastRect.x(), lastRect.y(), similarity, adaptiveSimilarityThreshold,
                (int)distance, timeSinceLastSeen);

            if (similarity <= adaptiveSimilarityThreshold) {
                log.info("✅ MATCHED: Person at ({}, {}) matched to existing track (similarity: {}, distance: {}px, track age: {}ms)",
                    faceRect.x(), faceRect.y(), similarity, (int)distance, trackDuration);
                return track;
            } else {
                log.debug("❌ Similarity too high: {} > {} - faces too different", similarity, adaptiveSimilarityThreshold);
            }
        }

        log.debug("🆕 NO MATCH: Creating new track for person at ({}, {})", faceRect.x(), faceRect.y());
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
        private final List<byte[]> gifFrames = new ArrayList<>();
        private final int maxGifFrames;
        private FaceObservation bestObservation; // Track the best scoring observation
        private byte[] bestImageBytes;
        private ScheduledFuture<?> pendingNotification; // Timeout notification future

        public TrackedPerson(
                Rect initialRect,
                byte[] initialFaceHash,
                long timestamp,
                double distanceScore,
                String personName,
                byte[] imageBytes,
                List<PersonDetection> allPeople,
                int maxGifFrames
        ) {
            this.firstSeen = timestamp;
            this.lastSeen = timestamp;
            this.maxGifFrames = maxGifFrames;
            FaceObservation observation = new FaceObservation(initialRect, initialFaceHash, timestamp, distanceScore, personName, allPeople);
            this.observations.add(observation);
            this.bestObservation = observation; // First is best by default
            this.bestImageBytes = imageBytes;
            storeGifFrame(imageBytes);

            // Log initial tracking state
            boolean isKnown = personName != null &&
                !"Unknown".equalsIgnoreCase(personName) &&
                !personName.toLowerCase().contains("unknown");
            log.debug("🆕 TRACKING STARTED: Frame 1, Identity='{}', Type={}, Distance={}",
                personName, isKnown ? "KNOWN" : "UNKNOWN", String.format("%.2f", distanceScore));
        }

        public void addObservation(Rect rect, byte[] faceHash, long timestamp, double distanceScore, String personName, byte[] imageBytes, List<PersonDetection> allPeople) {
            this.lastSeen = timestamp;
            FaceObservation observation = new FaceObservation(rect, faceHash, timestamp, distanceScore, personName, allPeople);
            this.observations.add(observation);
            storeGifFrame(imageBytes);

            // Calculate statistics for current tracking
            int totalFrames = observations.size();
            long knownCount = observations.stream()
                .filter(obs -> obs.personName != null &&
                    !"Unknown".equalsIgnoreCase(obs.personName) &&
                    !obs.personName.toLowerCase().contains("unknown"))
                .count();
            long unknownCount = totalFrames - knownCount;

            // Determine if current observation is known or unknown
            boolean isCurrentKnown = personName != null &&
                !"Unknown".equalsIgnoreCase(personName) &&
                !personName.toLowerCase().contains("unknown");

            // Log EVERY frame to track recognition results
            log.info("📸 FRAME {}: identity='{}' ({}), distance={}, Known count: {}/{}, Unknown count: {}/{}",
                totalFrames,
                personName,
                isCurrentKnown ? "KNOWN" : "UNKNOWN",
                String.format("%.2f", distanceScore),
                knownCount, totalFrames,
                unknownCount, totalFrames);

            // Log frame statistics every 30 frames
            if (totalFrames % 30 == 0) {
                log.info("📈 TRACKING PROGRESS: Frame {}: Current identity='{}', Known frames: {}/{} ({}%), Unknown frames: {}/{} ({}%)",
                    totalFrames, personName, knownCount, totalFrames,
                    String.format("%.1f", (knownCount * 100.0) / totalFrames),
                    unknownCount, totalFrames,
                    String.format("%.1f", (unknownCount * 100.0) / totalFrames));
            }

            // Update best observation if this one has lower distance (better match)
            if (bestObservation == null || distanceScore < bestObservation.distanceScore) {
                bestObservation = observation;
                bestImageBytes = imageBytes;
                log.debug("🎯 NEW BEST: Frame {} has new best distance score: {} for '{}'",
                    totalFrames, String.format("%.2f", distanceScore), personName);
            }
        }

        private void storeGifFrame(byte[] imageBytes) {
            if (imageBytes == null || imageBytes.length == 0) {
                return;
            }

            gifFrames.add(imageBytes);
            if (gifFrames.size() > maxGifFrames) {
                gifFrames.removeFirst();
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

        public List<PersonDetection> getBestAllPeople() {
            return (bestObservation != null && bestObservation.allPeople != null) ? bestObservation.allPeople :
                   (observations.isEmpty() ? new ArrayList<>() : observations.get(observations.size() - 1).allPeople);
        }

        public double getBestDistance() {
            return bestObservation != null ? bestObservation.distanceScore : 100.0;
        }

        public byte[] getBestImageBytes() {
            return bestImageBytes;
        }

        /**
         * Get all frame images from observations for GIF creation
         */
        public List<byte[]> getAllFrameImages() {
            return new ArrayList<>(gifFrames);
        }

        /**
         * Determine the most common identity across all observations.
         * If a known person appears in at least 3% of frames, they are recognized as that person.
         * This lower threshold helps identify people who turn their back to the camera.
         * @return IdentityResult containing the person name and count of frames they appeared in
         */
        public IdentityResult getMostCommonIdentity() {
            Map<String, Integer> identityCounts = new java.util.HashMap<>();
            int totalObservations = observations.size();

            // Count occurrences of each identity
            for (FaceObservation obs : observations) {
                String name = obs.personName != null ? obs.personName : "Unknown";
                identityCounts.put(name, identityCounts.getOrDefault(name, 0) + 1);
            }

            // Calculate statistics for logging
            int knownPersonFrames = 0;
            int unknownFrames = 0;
            for (Map.Entry<String, Integer> entry : identityCounts.entrySet()) {
                String name = entry.getKey();
                int count = entry.getValue();
                if ("Unknown".equalsIgnoreCase(name) || name.toLowerCase().contains("unknown")) {
                    unknownFrames += count;
                } else {
                    knownPersonFrames += count;
                }
            }

            // Log frame statistics
            log.info("📊 FRAME STATISTICS: Total frames: {}, Known person frames: {} ({}%), Unknown frames: {} ({}%)",
                totalObservations,
                knownPersonFrames, String.format("%.1f", (knownPersonFrames * 100.0) / totalObservations),
                unknownFrames, String.format("%.1f", (unknownFrames * 100.0) / totalObservations));

            // Log detailed breakdown by person
            StringBuilder breakdown = new StringBuilder("Frame breakdown by identity: ");
            for (Map.Entry<String, Integer> entry : identityCounts.entrySet()) {
                double percentage = (entry.getValue() * 100.0) / totalObservations;
                breakdown.append(String.format("%s: %d (%.1f%%), ", entry.getKey(), entry.getValue(), percentage));
            }
            log.info(breakdown.toString());

            // Minimum threshold: 5% of frames (reduced from 5% to handle back-turned cases)
            int minThreshold = Math.max(1, (int) Math.ceil(totalObservations * 0.05));

            // Find the most common KNOWN person (not Unknown) that meets the 3% threshold
            String bestKnownPerson = null;
            int bestKnownCount = 0;

            for (Map.Entry<String, Integer> entry : identityCounts.entrySet()) {
                String name = entry.getKey();
                int count = entry.getValue();

                // Skip "Unknown" when looking for known persons
                if (!"Unknown".equalsIgnoreCase(name) && !name.toLowerCase().contains("unknown")) {
                    if (count >= minThreshold && count > bestKnownCount) {
                        bestKnownCount = count;
                        bestKnownPerson = name;
                    }
                }
            }

            // If we found a known person with at least 5% of frames, return them
            if (bestKnownPerson != null) {
                double percentage = (bestKnownCount * 100.0) / totalObservations;
                log.info("✅ VERDICT: Determined identity: '{}' appeared in {} frames ({}%) out of {} total observations (threshold: {} frames = 5%)",
                    bestKnownPerson, bestKnownCount, String.format("%.1f", percentage), totalObservations, minThreshold);
                return new IdentityResult(bestKnownPerson, bestKnownCount);
            }

            // No known person reached 5% threshold, return Unknown with total frame count of other identities
            // Count all frames that are not Unknown
            int totalUnknownCount = 0;
            for (Map.Entry<String, Integer> entry : identityCounts.entrySet()) {
                String name = entry.getKey();
                if (!"Unknown".equalsIgnoreCase(name)) {
                    totalUnknownCount += entry.getValue();
                }
            }

            log.info("❌ VERDICT: No known person reached 5% threshold ({} frames). Unknown appeared in {} frames ({}%) out of {} observations - classifying as Unknown",
                minThreshold, totalUnknownCount, String.format("%.1f", (totalUnknownCount * 100.0) / totalObservations), totalObservations);

            return new IdentityResult("Unknown", totalUnknownCount);
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
            gifFrames.clear();
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
        private final double distanceScore; // Lower is better
        private final String personName; // Detected identity for this frame
        private final List<PersonDetection> allPeople; // All detected people with names and rectangles
    }

    /**
     * Represents a detected person with their rectangle and name
     */
    @Data
    @AllArgsConstructor
    public static class PersonDetection {
        private final String personName;
        private final Rect rect;
    }

    /**
     * Result of identity determination containing name and frame count
     */
    @Data
    @AllArgsConstructor
    public static class IdentityResult {
        private final String personName;
        private final int frameCount; // Number of frames this person appeared in
    }
}
