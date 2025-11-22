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

    // Minimum number of frames to track before sending notification (reduced from 5 to 3)
    private static final int MIN_TRACKING_FRAMES = 3;

    // Maximum time to track a face (10 seconds)
    private static final long MAX_TRACKING_TIME_MS = 10 * 1000;

    // Maximum distance between face positions to consider it the same face (pixels)
    private static final double MAX_POSITION_DISTANCE = 150.0;

    // Face similarity threshold for tracking (more lenient than detection)
    private static final int TRACKING_SIMILARITY_THRESHOLD = 25;

    // Minimum tracking duration before sending (1 second instead of 2)
    private static final long MIN_TRACKING_DURATION_MS = 1000;

    // Minimum movement required to consider face as real person (20px instead of 30)
    private static final double MIN_MOVEMENT_PIXELS = 20.0;

    // Maximum time gap between detections to keep tracking alive (5 seconds instead of 2)
    private static final long TRACK_TIMEOUT_MS = 5000;

    /**
     * Track an unknown face detection
     * Returns true if the face has been tracked long enough to send notification
     *
     * @param cameraName Name of the camera
     * @param faceRect Rectangle of the detected face
     * @param faceHash Hash of the face image for similarity comparison
     * @return true if should send notification, false if still tracking
     */
    public boolean trackUnknownFace(String cameraName, Rect faceRect, byte[] faceHash) {
        if (faceRect == null || faceHash == null) {
            return false; // Can't track without face data
        }

        long now = System.currentTimeMillis();
        List<TrackedFace> cameraTrackedFaces = trackedFaces.computeIfAbsent(cameraName, k -> new ArrayList<>());

        // Find if this face matches an existing tracked face
        TrackedFace matchedTrack = findMatchingTrackedFace(cameraTrackedFaces, faceRect, faceHash, now);

        if (matchedTrack != null) {
            // Update existing track - clone Rect to avoid issues if original is deallocated
            Rect clonedRect = new Rect(faceRect.x(), faceRect.y(), faceRect.width(), faceRect.height());
            matchedTrack.addObservation(clonedRect, faceHash, now);

            long trackingDuration = now - matchedTrack.firstSeen;
            int frameCount = matchedTrack.observations.size();

            log.debug("Camera '{}': Tracking unknown face - {} frames over {}ms (distance moved: {}px)",
                    cameraName, frameCount, trackingDuration, (int)matchedTrack.getTotalDistanceMoved());

            // Check if we've tracked long enough
            if (frameCount >= MIN_TRACKING_FRAMES && trackingDuration >= MIN_TRACKING_DURATION_MS) {
                // Check if face is actually moving (not static)
                double distanceMoved = matchedTrack.getTotalDistanceMoved();
                if (distanceMoved > MIN_MOVEMENT_PIXELS) {
                    log.info("Camera '{}': Unknown face tracked through {} frames over {}ms, moved {}px - sending notification",
                            cameraName, frameCount, trackingDuration, (int)distanceMoved);
                    // Remove from tracking and cleanup resources
                    cameraTrackedFaces.remove(matchedTrack);
                    matchedTrack.cleanup();
                    return true;
                } else {
                    log.debug("Camera '{}': Unknown face appears static (moved only {}px) - continuing to track",
                            cameraName, (int)distanceMoved);
                }
            }

            // Check if tracking expired (tracked too long without movement)
            if (trackingDuration > MAX_TRACKING_TIME_MS) {
                double distanceMoved = matchedTrack.getTotalDistanceMoved();
                if (distanceMoved < MIN_MOVEMENT_PIXELS) {
                    log.warn("Camera '{}': Unknown face tracked for {}ms but barely moved ({}px) - likely detection artifact, discarding",
                            cameraName, trackingDuration, (int)distanceMoved);
                    cameraTrackedFaces.remove(matchedTrack);
                    matchedTrack.cleanup();
                    return false; // Don't send - likely false positive
                } else {
                    log.info("Camera '{}': Unknown face tracking expired after {}ms, moved {}px - sending notification",
                            cameraName, trackingDuration, (int)distanceMoved);
                    cameraTrackedFaces.remove(matchedTrack);
                    matchedTrack.cleanup();
                    return true;
                }
            }

            return false; // Still tracking
        } else {
            // New face, start tracking - clone Rect to avoid issues if original is deallocated
            Rect clonedRect = new Rect(faceRect.x(), faceRect.y(), faceRect.width(), faceRect.height());
            TrackedFace newTrack = new TrackedFace(clonedRect, faceHash, now);
            cameraTrackedFaces.add(newTrack);
            log.debug("Camera '{}': Started tracking new unknown face at position ({}, {})",
                    cameraName, faceRect.x(), faceRect.y());
            return false; // Just started tracking
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
        for (TrackedFace track : tracks) {
            // Check if track is still recent (allow 5 second gaps instead of 2)
            if (currentTime - track.lastSeen > TRACK_TIMEOUT_MS) {
                continue; // Track is too old
            }

            // Check position distance
            Rect lastRect = track.getLatestRect();
            double distance = calculateDistance(faceRect, lastRect);
            if (distance > MAX_POSITION_DISTANCE) {
                continue; // Face moved too far
            }

            // Check face similarity
            int similarity = computeFaceHashSimilarity(faceHash, track.getLatestFaceHash());
            if (similarity <= TRACKING_SIMILARITY_THRESHOLD) {
                return track;
            }
        }
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

        public TrackedFace(Rect initialRect, byte[] initialFaceHash, long timestamp) {
            this.firstSeen = timestamp;
            this.lastSeen = timestamp;
            this.observations.add(new FaceObservation(initialRect, initialFaceHash, timestamp));
        }

        public void addObservation(Rect rect, byte[] faceHash, long timestamp) {
            this.lastSeen = timestamp;
            this.observations.add(new FaceObservation(rect, faceHash, timestamp));
        }

        public Rect getLatestRect() {
            return observations.get(observations.size() - 1).rect;
        }

        public byte[] getLatestFaceHash() {
            return observations.get(observations.size() - 1).faceHash;
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
    }
}
