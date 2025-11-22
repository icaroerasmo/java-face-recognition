package com.icaroerasmo.service;

import lombok.Data;
import lombok.extern.log4j.Log4j2;
import org.bytedeco.opencv.opencv_core.Rect;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Service to track unknown faces across multiple frames.
 * This helps determine if an unknown person is moving through the frame
 * or if it's just a detection artifact.
 */
@Log4j2
@Service
public class FaceTrackingService {

    // Track unknown faces per camera
    private final Map<String, List<TrackedFace>> trackedFaces = new ConcurrentHashMap<>();

    // Minimum number of frames to track before sending notification
    // Requires 5 detections of same person before sending to Telegram
    private static final int MIN_TRACKING_FRAMES = 5;

    // Maximum time to track a face (10 seconds)
    private static final long MAX_TRACKING_TIME_MS = 10 * 1000;

    // Maximum distance between face positions to consider it the same face (pixels)
    private static final double MAX_POSITION_DISTANCE = 150.0;

    // Face similarity threshold for tracking (extremely lenient)
    // Higher value = more lenient matching (allows more hash differences)
    // Set to 150 to allow almost any variation (hash check barely filters)
    // Only blocks if faces are COMPLETELY different (>150% difference)
    private static final int TRACKING_SIMILARITY_THRESHOLD = 150;

    // Minimum tracking duration before sending (1 second instead of 2)
    private static final long MIN_TRACKING_DURATION_MS = 1000;

    // Minimum movement required to consider face as real person (20px instead of 30)
    private static final double MIN_MOVEMENT_PIXELS = 20.0;

    // Maximum time gap between detections to keep tracking alive (5 seconds instead of 2)
    private static final long TRACK_TIMEOUT_MS = 5000;

    /**
     * Result of tracking containing best frame information and determined identity
     */
    @Data
    public static class TrackingResult {
        private final boolean shouldSend;
        private final String personName; // Determined identity (most common across frames)
        private final Rect bestRect;
        private final byte[] bestFaceHash;
        private final double bestConfidenceScore;
        private final byte[] bestImageBytes; // Image bytes of the best frame

        public static TrackingResult notReady() {
            return new TrackingResult(false, null, null, null, 0.0, null);
        }

        public static TrackingResult ready(String personName, Rect rect, byte[] faceHash, double score, byte[] imageBytes) {
            return new TrackingResult(true, personName, rect, faceHash, score, imageBytes);
        }
    }

    /**
     * Track a face detection (can be known or unknown)
     * Returns TrackingResult with most common identity and best frame if ready to send notification
     *
     * @param cameraName Name of the camera
     * @param personName Detected person name (or "Unknown")
     * @param faceRect Rectangle of the detected face
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
        List<TrackedFace> cameraTrackedFaces = trackedFaces.computeIfAbsent(cameraName, k -> new ArrayList<>());

        // Find if this face matches an existing tracked face
        TrackedFace matchedTrack = findMatchingTrackedFace(cameraTrackedFaces, faceRect, faceHash, now);

        if (matchedTrack != null) {
            // Update existing track - clone Rect to avoid issues if original is deallocated
            Rect clonedRect = new Rect(faceRect.x(), faceRect.y(), faceRect.width(), faceRect.height());
            matchedTrack.addObservation(clonedRect, faceHash, now, confidenceScore, personName, imageBytes);

            long trackingDuration = now - matchedTrack.firstSeen;
            int frameCount = matchedTrack.observations.size();

            log.debug("Camera '{}': Tracking face - {} frames over {}ms (distance moved: {}px, best score: {}, current: {})",
                    cameraName, frameCount, trackingDuration, (int)matchedTrack.getTotalDistanceMoved(),
                    String.format("%.2f", matchedTrack.getBestConfidenceScore()), personName);

            // Check if we've tracked long enough
            if (frameCount >= MIN_TRACKING_FRAMES && trackingDuration >= MIN_TRACKING_DURATION_MS) {
                // Check if face is actually moving (not static)
                double distanceMoved = matchedTrack.getTotalDistanceMoved();
                if (distanceMoved > MIN_MOVEMENT_PIXELS) {
                    // Determine the most common identity
                    String determinedIdentity = matchedTrack.getMostCommonIdentity();

                    log.info("Camera '{}': Face tracked through {} frames over {}ms, moved {}px, determined identity: '{}', best score: {} - sending notification",
                            cameraName, frameCount, trackingDuration, (int)distanceMoved, determinedIdentity,
                            String.format("%.2f", matchedTrack.getBestConfidenceScore()));

                    // Get best frame data before cleanup
                    Rect bestRect = matchedTrack.getBestRect();
                    byte[] bestFaceHash = matchedTrack.getBestFaceHash();
                    double bestScore = matchedTrack.getBestConfidenceScore();
                    byte[] bestImageBytes = matchedTrack.getBestImageBytes();

                    // Remove from tracking and cleanup resources
                    cameraTrackedFaces.remove(matchedTrack);
                    matchedTrack.cleanup();
                    return TrackingResult.ready(determinedIdentity, bestRect, bestFaceHash, bestScore, bestImageBytes);
                } else {
                    log.debug("Camera '{}': Face appears static (moved only {}px) - continuing to track",
                            cameraName, (int)distanceMoved);
                }
            }

            // Check if tracking expired (tracked too long without movement)
            if (trackingDuration > MAX_TRACKING_TIME_MS) {
                double distanceMoved = matchedTrack.getTotalDistanceMoved();
                if (distanceMoved < MIN_MOVEMENT_PIXELS) {
                    log.warn("Camera '{}': Face tracked for {}ms but barely moved ({}px) - likely detection artifact, discarding",
                            cameraName, trackingDuration, (int)distanceMoved);
                    cameraTrackedFaces.remove(matchedTrack);
                    matchedTrack.cleanup();
                    return TrackingResult.notReady(); // Don't send - likely false positive
                } else {
                    // Determine the most common identity
                    String determinedIdentity = matchedTrack.getMostCommonIdentity();

                    log.info("Camera '{}': Face tracking expired after {}ms, moved {}px, determined identity: '{}', best score: {} - sending notification",
                            cameraName, trackingDuration, (int)distanceMoved, determinedIdentity,
                            String.format("%.2f", matchedTrack.getBestConfidenceScore()));

                    // Get best frame data before cleanup
                    Rect bestRect = matchedTrack.getBestRect();
                    byte[] bestFaceHash = matchedTrack.getBestFaceHash();
                    double bestScore = matchedTrack.getBestConfidenceScore();
                    byte[] bestImageBytes = matchedTrack.getBestImageBytes();
                    cameraTrackedFaces.remove(matchedTrack);
                    matchedTrack.cleanup();
                    return TrackingResult.ready(determinedIdentity, bestRect, bestFaceHash, bestScore, bestImageBytes);
                }
            }

            return TrackingResult.notReady(); // Still tracking
        } else {
            // New face, start tracking - clone Rect to avoid issues if original is deallocated
            Rect clonedRect = new Rect(faceRect.x(), faceRect.y(), faceRect.width(), faceRect.height());
            TrackedFace newTrack = new TrackedFace(clonedRect, faceHash, now, confidenceScore, personName, imageBytes);
            cameraTrackedFaces.add(newTrack);
            log.debug("Camera '{}': Started tracking new face '{}' at position ({}, {}) with score {}",
                    cameraName, personName, faceRect.x(), faceRect.y(), String.format("%.2f", confidenceScore));
            return TrackingResult.notReady(); // Just started tracking
        }
    }

    /**
     * Cancel tracking for faces similar to a known person
     * This is called when a known person is detected to cancel any pending unknown tracks
     */
    public void cancelSimilarTracks(String cameraName, byte[] faceHash) {
        if (faceHash == null) {
            return;
        }

        List<TrackedFace> cameraTrackedFaces = trackedFaces.get(cameraName);
        if (cameraTrackedFaces == null || cameraTrackedFaces.isEmpty()) {
            return;
        }

        cameraTrackedFaces.removeIf(track -> {
            int similarity = computeFaceHashSimilarity(faceHash, track.getLatestFaceHash());
            if (similarity <= TRACKING_SIMILARITY_THRESHOLD) {
                log.info("Camera '{}': Cancelling tracked unknown face - matched with known person (similarity: {})",
                        cameraName, similarity);
                track.cleanup(); // Clean up native resources
                return true;
            }
            return false;
        });
    }

    /**
     * Find a tracked face that matches the current detection
     */
    private TrackedFace findMatchingTrackedFace(List<TrackedFace> tracks, Rect faceRect, byte[] faceHash, long currentTime) {
        log.debug("🔍 MATCHING: Checking {} existing tracks for face at ({}, {})", tracks.size(), faceRect.x(), faceRect.y());

        for (TrackedFace track : tracks) {
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
     * Cleanup old tracked faces
     */
    public void cleanupOldTracks() {
        long cutoffTime = System.currentTimeMillis() - MAX_TRACKING_TIME_MS;
        trackedFaces.values().forEach(tracks -> {
            tracks.removeIf(track -> {
                if (track.lastSeen < cutoffTime) {
                    track.cleanup(); // Clean up native resources
                    return true;
                }
                return false;
            });
        });
        log.debug("Cleaned up old face tracks");
    }

    /**
     * Represents a face being tracked across multiple frames
     */
    @Data
    private static class TrackedFace {
        private final long firstSeen;
        private long lastSeen;
        private final List<FaceObservation> observations = new ArrayList<>();
        private FaceObservation bestObservation; // Track the best scoring observation

        public TrackedFace(Rect initialRect, byte[] initialFaceHash, long timestamp, double confidenceScore, String personName, byte[] imageBytes) {
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

        public Rect getBestRect() {
            return bestObservation != null ? bestObservation.rect : getLatestRect();
        }

        public byte[] getBestFaceHash() {
            return bestObservation != null ? bestObservation.faceHash : getLatestFaceHash();
        }

        public double getBestConfidenceScore() {
            return bestObservation != null ? bestObservation.confidenceScore : 100.0;
        }

        public byte[] getBestImageBytes() {
            return bestObservation != null ? bestObservation.imageBytes : null;
        }

        /**
         * Determine the most common identity across all observations
         * @return The person name that appears most frequently (or "Unknown" if tied/most common)
         */
        public String getMostCommonIdentity() {
            Map<String, Integer> identityCounts = new java.util.HashMap<>();

            // Count occurrences of each identity
            for (FaceObservation obs : observations) {
                String name = obs.personName != null ? obs.personName : "Unknown";
                identityCounts.put(name, identityCounts.getOrDefault(name, 0) + 1);
            }

            // Find the most common identity
            String mostCommon = "Unknown";
            int maxCount = 0;
            int unknownCount = identityCounts.getOrDefault("Unknown", 0);

            for (Map.Entry<String, Integer> entry : identityCounts.entrySet()) {
                if (entry.getValue() > maxCount) {
                    maxCount = entry.getValue();
                    mostCommon = entry.getKey();
                }
            }

            // If Unknown appears at least as much as the winner, prefer Unknown
            // This is conservative - only identify as known if clearly dominant
            if (!mostCommon.equals("Unknown") && unknownCount >= maxCount * 0.4) {
                log.debug("Identity uncertain: '{}' ({} times) vs Unknown ({} times) - reporting as Unknown",
                    mostCommon, maxCount, unknownCount);
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
         * Clean up native resources (deallocate Rect objects)
         */
        public void cleanup() {
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
