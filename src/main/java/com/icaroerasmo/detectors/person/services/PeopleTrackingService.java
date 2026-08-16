package com.icaroerasmo.detectors.person.services;

import com.icaroerasmo.properties.StreamsProperties;
import com.icaroerasmo.service.GifCreationService;
import com.icaroerasmo.service.TelegramPublisherService;
import com.icaroerasmo.utils.MatUtil;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.extern.log4j.Log4j2;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Size;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.*;
import jakarta.annotation.PreDestroy;

import static com.icaroerasmo.utils.FaceHashUtils.computeSimilarity;

/**
 * Service to track people (both known and unknown) across multiple frames.
 * This helps prevent duplicate notifications for the same person and ensures
 * notifications are only sent for legitimate detections with movement.
 */
@Log4j2
@Service
public class PeopleTrackingService {

    private static final String UNKNOWN_IDENTITY = "Unknown";
    private static final int GIF_FRAME_MAX_WIDTH = 640;
    private static final int MIN_KNOWN_FRAMES_FOR_IDENTITY = 3;

    private final TelegramPublisherService telegramPublisherService;
    private final DetectionHistoryService detectionHistoryService;
    private final MatUtil matUtil;
    private final GifCreationService gifCreationService;
    private final StreamsProperties streamsProperties;
    private final ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(4);

    // Track people per camera
    private final Map<String, List<TrackedPerson>> trackedPeople = new ConcurrentHashMap<>();

    // Timeout for sending notification if person not detected anymore (5 seconds for better consolidation)
    private static final long NOTIFICATION_TIMEOUT_MS = 5000;

    public PeopleTrackingService(TelegramPublisherService telegramPublisherService,
                               DetectionHistoryService detectionHistoryService,
                               MatUtil matUtil,
                               GifCreationService gifCreationService,
                               StreamsProperties streamsProperties) {
        this.telegramPublisherService = telegramPublisherService;
        this.detectionHistoryService = detectionHistoryService;
        this.matUtil = matUtil;
        this.gifCreationService = gifCreationService;
        this.streamsProperties = streamsProperties;
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

    // Maximum time to track a person (20 seconds to handle people turning/walking away)
    private static final long MAX_TRACKING_TIME_MS = 20 * 1000;

    // Maximum distance between positions to consider it the same person (pixels)
    // Increased to 500px to track people walking away or across the frame
    private static final double MAX_POSITION_DISTANCE = 500.0;

    // Perceptual-hash distance threshold (0 is identical, 100 is completely different).
    private static final int TRACKING_SIMILARITY_THRESHOLD = 40;

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
     * For continued tracks (with confirmed identity), the track is reset instead of removed.
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
                IdentityResult identityResult = determineIdentity(track, cameraName);
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

                // Check if this is a continued track (has confirmed identity from previous finalization)
                // If so, reset instead of removing to preserve identity for re-identification
                if (track.getConfirmedIdentity() != null && !isUnknownIdentity(track.getConfirmedIdentity())) {
                    log.info("Camera '{}': Resetting continued track for '{}' after timeout (preserving identity)",
                        cameraName, track.getConfirmedIdentity());
                    long resetTimestamp = System.currentTimeMillis();
                    track.resetTrack(track.getConfirmedIdentity(), resetTimestamp);
                    // Reschedule timeout for the reset track
                    scheduleTimeoutNotification(track, cameraName, cameraTrackedPeople);
                } else {
                    // Original behavior: remove from tracking and cleanup
                    cameraTrackedPeople.remove(track);
                    track.cleanup();
                }

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
        Mat originalImg = null;
        Mat imageBufferMat = null;
        BytePointer imagePointer = null;
        BytePointer buf = null;
        BytePointer jpgExt = null;
        try {
            log.info("🔔 SENDING NOTIFICATION: camera='{}', identity='{}', score={}, people count={}, identityFrames={}, totalFrames={}, totalFrameImages={}",
                cameraName, determinedIdentity, String.format("%.2f", bestScore),
                allPeople != null ? allPeople.size() : 0, identityFrameCount, totalTrackedFrames,
                allFrameImages != null ? allFrameImages.size() : 0);

            if (bestImageBytes == null || bestImageBytes.length == 0) {
                log.error("Cannot send notification: image bytes is null or empty");
                return;
            }

            boolean isUnknown = isUnknownIdentity(determinedIdentity);

            // Convert image bytes to Mat so we can draw on it
            imagePointer = new org.bytedeco.javacpp.BytePointer(bestImageBytes);
            imageBufferMat = new org.bytedeco.opencv.opencv_core.Mat(imagePointer);
            originalImg = org.bytedeco.opencv.global.opencv_imgcodecs.imdecode(
                imageBufferMat,
                org.bytedeco.opencv.global.opencv_imgcodecs.IMREAD_COLOR
            );

            if (originalImg.empty()) {
                log.error("Failed to decode image bytes to Mat");
                return;
            }

            // Clone the image and draw rectangle on it
            annotatedImg = originalImg.clone();

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
            buf = new org.bytedeco.javacpp.BytePointer();
            jpgExt = new org.bytedeco.javacpp.BytePointer(".jpg");
            org.bytedeco.opencv.global.opencv_imgcodecs.imencode(jpgExt, annotatedImg, buf);
            byte[] annotatedImageBytes = new byte[(int) buf.limit()];
            buf.get(annotatedImageBytes);

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
            if (imagePointer != null) {
                imagePointer.deallocate();
            }
            if (imageBufferMat != null) {
                imageBufferMat.deallocate();
            }
            if (buf != null) {
                buf.deallocate();
            }
            if (jpgExt != null) {
                jpgExt.deallocate();
            }
            if (originalImg != null) {
                matUtil.releaseResources(originalImg);
            }
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
    public TrackingResult trackFace(
            String cameraName,
            String personName,
            Rect faceRect,
            byte[] faceHash,
            double distance,
            byte[] imageBytes,
            List<PersonDetection> allPeople,
            boolean hasVisibleFace
    ) {
        if (faceRect == null || faceHash == null || imageBytes == null) {
            return TrackingResult.notReady(); // Can't track without face data
        }

        long now = System.currentTimeMillis();
        List<TrackedPerson> cameraTrackedPeople = trackedPeople.computeIfAbsent(cameraName, k -> new ArrayList<>());

        // Find if this person matches an existing tracked person
        TrackedPerson matchedTrack = findMatchingTrackedPerson(cameraTrackedPeople, faceRect, faceHash, now, personName);

        if (matchedTrack != null) {
            // Update existing track - clone Rect to avoid issues if original is deallocated
            Rect clonedRect = new Rect(faceRect.x(), faceRect.y(), faceRect.width(), faceRect.height());
            matchedTrack.addObservation(clonedRect, faceHash, now, distance, personName, imageBytes, allPeople, hasVisibleFace);

            long trackingDuration = now - matchedTrack.firstSeen;
            int frameCount = matchedTrack.observations.size();

            log.debug("Camera '{}': Tracking person - {} frames over {}ms (distance moved: {}px, best score: {}, current: {})",
                    cameraName, frameCount, trackingDuration, (int)matchedTrack.getTotalDistanceMoved(),
                    String.format("%.2f", matchedTrack.getBestDistance()), personName);

            // Cancel any existing timeout notification and schedule a new one
            // This ensures notification is sent if person disappears before reaching threshold
            matchedTrack.cancelPendingNotification();
            scheduleTimeoutNotification(matchedTrack, cameraName, cameraTrackedPeople);

            double distanceMoved = matchedTrack.getTotalDistanceMoved();

            if (frameCount >= getMaxTrackingFrames()) {
                log.info("Camera '{}': Person tracking reached max frame cap of {} after {}ms (moved {}px) - finalizing track",
                    cameraName, getMaxTrackingFrames(), trackingDuration, (int) distanceMoved);
                if (distanceMoved < MIN_MOVEMENT_PIXELS) {
                    return discardTrack(cameraTrackedPeople, matchedTrack, cameraName, trackingDuration, distanceMoved,
                        "max frame cap without sufficient movement");
                }
                return finalizeTrack(cameraTrackedPeople, matchedTrack, cameraName, trackingDuration, distanceMoved,
                    "max frame cap reached");
            }

            // Check if we've tracked long enough
            if (frameCount >= getMinTrackingFrames() && trackingDuration >= MIN_TRACKING_DURATION_MS) {
                // Check if person is actually moving (not static)
                if (distanceMoved > MIN_MOVEMENT_PIXELS) {
                    return finalizeTrack(cameraTrackedPeople, matchedTrack, cameraName, trackingDuration, distanceMoved,
                        "ready to send notification");
                } else {
                    log.debug("Camera '{}': Person appears static (moved only {}px) - continuing to track",
                            cameraName, (int)distanceMoved);
                }
            }

            // Check if tracking expired (tracked too long without movement)
            if (trackingDuration > MAX_TRACKING_TIME_MS) {
                if (distanceMoved < MIN_MOVEMENT_PIXELS) {
                    return discardTrack(cameraTrackedPeople, matchedTrack, cameraName, trackingDuration, distanceMoved,
                        "tracking timeout without sufficient movement");
                } else {
                    return finalizeTrack(cameraTrackedPeople, matchedTrack, cameraName, trackingDuration, distanceMoved,
                        "tracking timeout reached");
                }
            }

            return TrackingResult.notReady(); // Still tracking
        } else {
            if (!isUnknownIdentity(personName) && hasVisibleFace) {
                removeOverlappingUnknownTracks(cameraTrackedPeople, faceRect, cameraName, personName);
            }

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
                gifCreationService.getMaxGifFrames(),
                hasVisibleFace
            );
            cameraTrackedPeople.add(newTrack);

            // Schedule timeout notification in case person disappears
            scheduleTimeoutNotification(newTrack, cameraName, cameraTrackedPeople);

            log.debug("Camera '{}': Started tracking new person '{}' at position ({}, {}) with score {} (timeout notification scheduled)",
                    cameraName, personName, faceRect.x(), faceRect.y(), String.format("%.2f", distance));
            return TrackingResult.notReady(); // Just started tracking
        }
    }

    private void removeOverlappingUnknownTracks(
            List<TrackedPerson> cameraTrackedPeople,
            Rect faceRect,
            String cameraName,
            String personName
    ) {
        List<TrackedPerson> tracksToRemove = new ArrayList<>();

        for (TrackedPerson track : cameraTrackedPeople) {
            if (!track.isUnknownTrack()) {
                continue;
            }

            Rect trackedRect = track.getLatestRect();
            if (trackedRect == null) {
                continue;
            }

            if (containsFaceCenter(trackedRect, faceRect) || intersects(trackedRect, faceRect)) {
                tracksToRemove.add(track);
            }
        }

        for (TrackedPerson track : tracksToRemove) {
            cameraTrackedPeople.remove(track);
            track.cleanup();
            log.info("Removed overlapping unknown track on camera '{}' after recognized face '{}' appeared",
                cameraName, personName);
        }
    }

    private boolean containsFaceCenter(Rect outerRect, Rect innerRect) {
        int centerX = innerRect.x() + (innerRect.width() / 2);
        int centerY = innerRect.y() + (innerRect.height() / 2);
        return centerX >= outerRect.x()
            && centerX <= outerRect.x() + outerRect.width()
            && centerY >= outerRect.y()
            && centerY <= outerRect.y() + outerRect.height();
    }

    private boolean intersects(Rect rect1, Rect rect2) {
        int left = Math.max(rect1.x(), rect2.x());
        int top = Math.max(rect1.y(), rect2.y());
        int right = Math.min(rect1.x() + rect1.width(), rect2.x() + rect2.width());
        int bottom = Math.min(rect1.y() + rect1.height(), rect2.y() + rect2.height());
        return right > left && bottom > top;
    }


    /**
     * Find a tracked person that matches the current detection.
     * Uses adaptive matching - more lenient for established tracks to handle people turning around.
     * Also supports identity-based matching for tracks with confirmed identities.
     */
    private TrackedPerson findMatchingTrackedPerson(List<TrackedPerson> tracks, Rect faceRect, byte[] faceHash, long currentTime, String personName) {
        log.debug("🔍 MATCHING: Checking {} existing tracks for person at ({}, {}) with identity '{}'",
            tracks.size(), faceRect.x(), faceRect.y(), personName);

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

            // Check if this track has a confirmed identity from a previous finalization
            String confirmedIdentity = track.getConfirmedIdentity();
            boolean hasConfirmedIdentity = confirmedIdentity != null && !isUnknownIdentity(confirmedIdentity);

            // Adaptive similarity threshold based on tracking duration
            // For established tracks (>2 seconds), be more lenient about face similarity
            // This helps track people who turn their back or change angles
            long trackDuration = currentTime - track.firstSeen;
            int adaptiveSimilarityThreshold = TRACKING_SIMILARITY_THRESHOLD;

            if (trackDuration > 2000) { // More than 2 seconds of tracking
                // For close distances (<150px), be very lenient - probably same person
                if (distance < 150) {
                    adaptiveSimilarityThreshold = 55;
                } else {
                    adaptiveSimilarityThreshold = 45;
                }
                log.debug("📈 Using adaptive threshold {} (track age: {}ms, distance: {}px)",
                    adaptiveSimilarityThreshold, trackDuration, (int)distance);
            }

            // Extra leniency for tracks with confirmed identities
            // These tracks have already been identified, so we want to keep them alive
            if (hasConfirmedIdentity) {
                // If the current detection matches the confirmed identity, be very lenient
                if (!isUnknownIdentity(personName) && personName.equals(confirmedIdentity)) {
                    // Same identity confirmed - use very lenient threshold
                    adaptiveSimilarityThreshold = Math.max(adaptiveSimilarityThreshold, 65);
                    log.debug("📈 Using extra lenient threshold {} for confirmed identity '{}' match",
                        adaptiveSimilarityThreshold, confirmedIdentity);
                } else if (isUnknownIdentity(personName)) {
                    // Unknown face but track has confirmed identity - be lenient if position is close
                    if (distance < 200) {
                        adaptiveSimilarityThreshold = Math.max(adaptiveSimilarityThreshold, 60);
                        log.debug("📈 Using lenient threshold {} for confirmed identity '{}' (unknown face, close position)",
                            adaptiveSimilarityThreshold, confirmedIdentity);
                    }
                }
            }

            // Check face hash similarity
            int similarity = computeSimilarity(faceHash, track.getLatestFaceHash());
            log.debug("📊 Track at ({}, {}) similarity: {} (threshold: {}), distance: {}px, age: {}ms, confirmedIdentity: {}",
                lastRect.x(), lastRect.y(), similarity, adaptiveSimilarityThreshold,
                (int)distance, timeSinceLastSeen, confirmedIdentity);

            // SAFEGUARD: Reset confirmed identity ONLY if a DIFFERENT known person appears
            // This prevents different people from inheriting identity when appearing in similar positions
            // BUT does NOT reset when face is obscured/unknown (same person lowering face)
            if (hasConfirmedIdentity && similarity > 50 && !isUnknownIdentity(personName) && !personName.equals(confirmedIdentity)) {
                log.info("🔄 IDENTITY RESET: Track at ({}, {}) had confirmed identity '{}' but detected different person '{}' (similarity: {}). Resetting identity.",
                    lastRect.x(), lastRect.y(), confirmedIdentity, personName, similarity);
                track.resetTrack(null, currentTime);
                // Continue to normal matching logic without the confirmed identity advantage
                hasConfirmedIdentity = false;
                confirmedIdentity = null;
                // Reset adaptive threshold to base value
                adaptiveSimilarityThreshold = TRACKING_SIMILARITY_THRESHOLD;
            }

            if (similarity <= adaptiveSimilarityThreshold) {
                log.info("✅ MATCHED: Person at ({}, {}) matched to existing track (similarity: {}, distance: {}px, track age: {}ms, confirmedIdentity: {})",
                    faceRect.x(), faceRect.y(), similarity, (int)distance, trackDuration, confirmedIdentity);
                return track;
            } else {
                log.debug("❌ Similarity too high: {} > {} - faces too different", similarity, adaptiveSimilarityThreshold);
            }
        }

        log.debug("🆕 NO MATCH: Creating new track for person at ({}, {})", faceRect.x(), faceRect.y());
        return null;
    }

    private TrackingResult finalizeTrack(
            List<TrackedPerson> cameraTrackedPeople,
            TrackedPerson track,
            String cameraName,
            long trackingDuration,
            double distanceMoved,
            String completionReason
    ) {
        int frameCount = track.observations.size();
        IdentityResult identityResult = determineIdentity(track, cameraName);
        String determinedIdentity = identityResult.getPersonName();
        int identityFrameCount = identityResult.getFrameCount();

        log.info("Camera '{}': Person tracking completed ({}) after {} frames over {}ms, moved {}px, determined identity: '{}', best score: {}",
            cameraName, completionReason, frameCount, trackingDuration, (int) distanceMoved, determinedIdentity,
            String.format("%.2f", track.getBestDistance()));

        track.cancelPendingNotification();

        byte[] bestFaceHash = track.getBestFaceHash();
        double bestScore = track.getBestDistance();
        byte[] bestImageBytes = track.getBestImageBytes();
        List<PersonDetection> bestAllPeople = track.getBestAllPeople();
        List<byte[]> allFrameImages = track.getAllFrameImages();

        sendNotificationNow(cameraName, determinedIdentity, bestScore, bestImageBytes, bestFaceHash, bestAllPeople,
            identityFrameCount, frameCount, allFrameImages);

        // Reset track instead of removing it - preserve identity for quick re-identification
        long resetTimestamp = System.currentTimeMillis();
        track.resetTrack(determinedIdentity, resetTimestamp);

        // Reschedule timeout notification for the continued track
        scheduleTimeoutNotification(track, cameraName, cameraTrackedPeople);

        log.info("Camera '{}': Track continued for '{}' after finalization - ready for re-identification",
            cameraName, determinedIdentity);

        return new TrackingResult(true, determinedIdentity, bestFaceHash, bestScore, bestImageBytes, bestAllPeople, identityFrameCount);
    }

    private TrackingResult discardTrack(
            List<TrackedPerson> cameraTrackedPeople,
            TrackedPerson track,
            String cameraName,
            long trackingDuration,
            double distanceMoved,
            String discardReason
    ) {
        log.warn("Camera '{}': Discarding track ({}) after {} frames over {}ms with only {}px of movement",
            cameraName, discardReason, track.observations.size(), trackingDuration, (int) distanceMoved);
        cameraTrackedPeople.remove(track);
        track.cleanup();
        return TrackingResult.notReady();
    }

    private IdentityResult determineIdentity(TrackedPerson track, String cameraName) {
        IdentityResult identityResult = track.getMostCommonIdentity();
        if (!isUnknownIdentity(identityResult.getPersonName())) {
            return identityResult;
        }

        IdentityResult consistentCandidate = track.getConsistentKnownCandidate();
        if (consistentCandidate != null
                && consistentCandidate.getFrameCount() >= 2
                && track.observations.size() <= MIN_KNOWN_FRAMES_FOR_IDENTITY
                && detectionHistoryService.wasKnownPersonDetectedRecently(
                    cameraName,
                    consistentCandidate.getPersonName()
                )) {
            log.info("✅ VERDICT: Continuing recent confirmed identity '{}' across a short track fragment with {} agreeing frame(s)",
                consistentCandidate.getPersonName(), consistentCandidate.getFrameCount());
            return consistentCandidate;
        }

        return identityResult;
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
        private long firstSeen;
        private long lastSeen;
        private final List<FaceObservation> observations = new ArrayList<>();
        private final List<byte[]> gifFrames = new ArrayList<>();
        private final int maxGifFrames;
        private FaceObservation bestObservation; // Track the best scoring observation
        private byte[] bestImageBytes;
        private ScheduledFuture<?> pendingNotification; // Timeout notification future
        private String confirmedIdentity; // Persisted identity across track resets

        public TrackedPerson(
                Rect initialRect,
                byte[] initialFaceHash,
                long timestamp,
                double distanceScore,
                String personName,
                byte[] imageBytes,
                List<PersonDetection> allPeople,
                int maxGifFrames,
                boolean hasVisibleFace
        ) {
            this.firstSeen = timestamp;
            this.lastSeen = timestamp;
            this.maxGifFrames = maxGifFrames;
            FaceObservation observation = new FaceObservation(
                initialRect,
                initialFaceHash,
                timestamp,
                distanceScore,
                personName,
                clonePersonDetections(allPeople),
                hasVisibleFace
            );
            this.observations.add(observation);
            this.bestObservation = observation; // First is best by default
            this.bestImageBytes = imageBytes;
            storeGifFrame(imageBytes);

            // Log initial tracking state
            boolean isKnown = !isUnknownIdentity(personName);
            log.debug("🆕 TRACKING STARTED: Frame 1, Identity='{}', Type={}, Distance={}",
                personName, isKnown ? "KNOWN" : "UNKNOWN", String.format("%.2f", distanceScore));
        }

        /**
         * Reset the track after finalization while preserving confirmed identity.
         * This allows the same person to be quickly re-identified in subsequent frames
         * without losing tracking state.
         *
         * @param identity The confirmed identity to preserve
         * @param resetTimestamp The timestamp to use as the new firstSeen
         */
        public void resetTrack(String identity, long resetTimestamp) {
            // Store confirmed identity before clearing
            this.confirmedIdentity = identity;

            // Clear old observations (keep last 2 for position matching)
            int keepCount = Math.min(2, observations.size());
            if (keepCount > 0) {
                List<FaceObservation> recentObservations = new ArrayList<>(
                    observations.subList(observations.size() - keepCount, observations.size())
                );
                observations.clear();
                observations.addAll(recentObservations);
            } else {
                observations.clear();
            }

            // Clear gif frames (old ones are already used in the generated GIF)
            gifFrames.clear();

            // Reset timing
            this.firstSeen = resetTimestamp;
            this.lastSeen = resetTimestamp;

            // Reset best observation to null (will be set by next observation)
            this.bestObservation = null;
            this.bestImageBytes = null;

            log.info("🔄 TRACK RESET: Identity='{}' preserved, observations cleared to {}, gif frames cleared",
                identity, observations.size());
        }

        /**
         * Get the confirmed identity if this track was previously finalized.
         * @return The confirmed identity, or null if not yet finalized
         */
        public String getConfirmedIdentity() {
            return confirmedIdentity;
        }

        public void addObservation(
                Rect rect,
                byte[] faceHash,
                long timestamp,
                double distanceScore,
                String personName,
                byte[] imageBytes,
                List<PersonDetection> allPeople,
                boolean hasVisibleFace
        ) {
            this.lastSeen = timestamp;
            FaceObservation observation = new FaceObservation(
                rect,
                faceHash,
                timestamp,
                distanceScore,
                personName,
                clonePersonDetections(allPeople),
                hasVisibleFace
            );
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
            boolean isCurrentKnown = !isUnknownIdentity(personName);

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

            // Prefer observations with a visible face; otherwise fall back to lowest distance.
            boolean shouldReplaceBest = bestObservation == null
                || (hasVisibleFace && !bestObservation.hasVisibleFace)
                || (hasVisibleFace == bestObservation.hasVisibleFace && distanceScore < bestObservation.distanceScore);

            if (shouldReplaceBest) {
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

            gifFrames.add(createGifFrameSnapshot(imageBytes));
            if (gifFrames.size() > maxGifFrames) {
                gifFrames.removeFirst();
            }
        }

        private byte[] createGifFrameSnapshot(byte[] imageBytes) {
            BytePointer imagePointer = null;
            BytePointer jpgExt = null;
            BytePointer outputBuffer = null;
            Mat encodedInputMat = null;
            Mat decodedFrame = null;
            Mat resizedFrame = null;
            Size resizeSize = null;

            try {
                imagePointer = new BytePointer(imageBytes);
                encodedInputMat = new org.bytedeco.opencv.opencv_core.Mat(imagePointer);
                decodedFrame = org.bytedeco.opencv.global.opencv_imgcodecs.imdecode(
                    encodedInputMat,
                    org.bytedeco.opencv.global.opencv_imgcodecs.IMREAD_COLOR
                );

                if (decodedFrame == null || decodedFrame.empty()) {
                    return imageBytes;
                }

                if (decodedFrame.cols() <= GIF_FRAME_MAX_WIDTH) {
                    return imageBytes;
                }

                int scaledHeight = Math.max(1, (int) Math.round(
                    decodedFrame.rows() * (GIF_FRAME_MAX_WIDTH / (double) decodedFrame.cols())
                ));

                resizedFrame = new Mat();
                resizeSize = new Size(GIF_FRAME_MAX_WIDTH, scaledHeight);
                org.bytedeco.opencv.global.opencv_imgproc.resize(
                    decodedFrame,
                    resizedFrame,
                    resizeSize
                );

                if (resizedFrame.empty()) {
                    return imageBytes;
                }

                jpgExt = new BytePointer(".jpg");
                outputBuffer = new BytePointer();
                org.bytedeco.opencv.global.opencv_imgcodecs.imencode(jpgExt, resizedFrame, outputBuffer);

                byte[] resizedBytes = new byte[(int) outputBuffer.limit()];
                outputBuffer.get(resizedBytes);
                return resizedBytes;
            } catch (Exception e) {
                log.debug("Failed to compress tracking frame for GIF storage, using original bytes: {}", e.getMessage());
                return imageBytes;
            } finally {
                if (imagePointer != null) {
                    imagePointer.deallocate();
                }
                if (jpgExt != null) {
                    jpgExt.deallocate();
                }
                if (outputBuffer != null) {
                    outputBuffer.deallocate();
                }
                if (encodedInputMat != null) {
                    encodedInputMat.deallocate();
                }
                if (resizeSize != null) {
                    resizeSize.deallocate();
                }
                MatUtil.deallocateMats(decodedFrame, resizedFrame);
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

        public boolean isUnknownTrack() {
            return observations.stream().allMatch(observation -> isUnknownIdentity(observation.personName));
        }

        /**
         * Get all frame images from observations for GIF creation
         */
        public List<byte[]> getAllFrameImages() {
            return new ArrayList<>(gifFrames);
        }

        /**
         * Determine the most common identity across all observations.
         * Only non-unknown observations participate in the final vote.
         * @return IdentityResult containing the person name and count of frames they appeared in
         */
        public IdentityResult getMostCommonIdentity() {
            int totalObservations = observations.size();
            if (totalObservations < MIN_KNOWN_FRAMES_FOR_IDENTITY) {
                log.info("❌ VERDICT: Only {} captured frame(s) in track. Minimum required is {}, classifying as '{}'.",
                    totalObservations, MIN_KNOWN_FRAMES_FOR_IDENTITY, UNKNOWN_IDENTITY);
                return new IdentityResult(UNKNOWN_IDENTITY, totalObservations);
            }

            List<FaceObservation> usefulObservations = observations.stream()
                .filter(observation -> !isUnknownIdentity(observation.personName))
                .toList();
            int usefulFrames = usefulObservations.size();
            int unknownFrames = totalObservations - usefulFrames;

            // Calculate statistics for logging
            log.info("📊 FRAME STATISTICS: Total frames: {}, Known person frames: {} ({}%), Unknown frames: {} ({}%)",
                totalObservations,
                usefulFrames, String.format("%.1f", (usefulFrames * 100.0) / totalObservations),
                unknownFrames, String.format("%.1f", (unknownFrames * 100.0) / totalObservations));

            if (usefulFrames < MIN_KNOWN_FRAMES_FOR_IDENTITY) {
                log.info("❌ VERDICT: Only {} useful frame(s) out of {} total frames. Minimum required is {}, classifying as '{}'.",
                    usefulFrames, totalObservations, MIN_KNOWN_FRAMES_FOR_IDENTITY, UNKNOWN_IDENTITY);
                return new IdentityResult(UNKNOWN_IDENTITY, totalObservations);
            }

            Map<String, Integer> identityCounts = new java.util.HashMap<>();
            for (FaceObservation obs : usefulObservations) {
                String name = obs.personName != null ? obs.personName : UNKNOWN_IDENTITY;
                identityCounts.put(name, identityCounts.getOrDefault(name, 0) + 1);
            }

            // Log detailed breakdown by person
            StringBuilder breakdown = new StringBuilder("Frame breakdown by identity: ");
            for (Map.Entry<String, Integer> entry : identityCounts.entrySet()) {
                double percentage = (entry.getValue() * 100.0) / usefulFrames;
                breakdown.append(String.format("%s: %d (%.1f%%), ", entry.getKey(), entry.getValue(), percentage));
            }
            log.info(breakdown.toString());

            String winningIdentity = UNKNOWN_IDENTITY;
            int winningCount = 0;
            boolean tiedWinner = false;

            for (Map.Entry<String, Integer> entry : identityCounts.entrySet()) {
                String name = entry.getKey();
                int count = entry.getValue();

                if (count > winningCount) {
                    winningIdentity = name;
                    winningCount = count;
                    tiedWinner = false;
                } else if (count == winningCount && !Objects.equals(name, winningIdentity)) {
                    tiedWinner = true;
                }
            }

            if (tiedWinner) {
                log.info("⚠️ VERDICT: Identity pool tied at {} frames. Falling back to '{}' to avoid guessing.",
                    winningCount, UNKNOWN_IDENTITY);
                return new IdentityResult(UNKNOWN_IDENTITY, usefulFrames);
            }

            if (winningCount < MIN_KNOWN_FRAMES_FOR_IDENTITY) {
                log.info("❌ VERDICT: Winning identity '{}' only has {} non-unknown frame(s). Minimum required is {}, classifying as '{}'.",
                    winningIdentity, winningCount, MIN_KNOWN_FRAMES_FOR_IDENTITY, UNKNOWN_IDENTITY);
                return new IdentityResult(UNKNOWN_IDENTITY, totalObservations);
            }

            double percentage = (winningCount * 100.0) / usefulFrames;
            log.info("✅ VERDICT: Identity pool winner is '{}' with {} useful frames ({}%) out of {} useful / {} total observations",
                winningIdentity, winningCount, String.format("%.1f", percentage), usefulFrames, totalObservations);
            return new IdentityResult(winningIdentity, winningCount);
        }

        public IdentityResult getConsistentKnownCandidate() {
            String candidate = null;
            int count = 0;

            for (FaceObservation observation : observations) {
                if (isUnknownIdentity(observation.personName)) {
                    continue;
                }
                if (candidate == null) {
                    candidate = observation.personName;
                } else if (!candidate.equals(observation.personName)) {
                    return null;
                }
                count++;
            }

            return candidate != null ? new IdentityResult(candidate, count) : null;
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
                releasePersonDetections(observation.allPeople);
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
        private final boolean hasVisibleFace;
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

    private static boolean isUnknownIdentity(String personName) {
        return personName == null || UNKNOWN_IDENTITY.equalsIgnoreCase(personName) || personName.toLowerCase().contains("unknown");
    }

    private static List<PersonDetection> clonePersonDetections(List<PersonDetection> detections) {
        if (detections == null || detections.isEmpty()) {
            return List.of();
        }

        List<PersonDetection> clones = new ArrayList<>(detections.size());
        for (PersonDetection detection : detections) {
            if (detection == null) {
                continue;
            }
            clones.add(new PersonDetection(detection.getPersonName(), MatUtil.cloneRect(detection.getRect())));
        }
        return clones;
    }

    private static void releasePersonDetections(List<PersonDetection> detections) {
        if (detections == null || detections.isEmpty()) {
            return;
        }

        List<Rect> rects = new ArrayList<>(detections.size());
        for (PersonDetection detection : detections) {
            if (detection != null && detection.getRect() != null) {
                rects.add(detection.getRect());
            }
        }
        MatUtil.deallocateRects(rects);
    }

    private int getMinTrackingFrames() {
        return Math.max(5, streamsProperties.getProcessingFps() * 5);
    }

    private int getMaxTrackingFrames() {
        return Math.max(3, streamsProperties.getTrackingMaxFrames());
    }
}
