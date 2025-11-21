package com.icaroerasmo.service;

import lombok.extern.log4j.Log4j2;
import org.springframework.stereotype.Service;

import java.security.MessageDigest;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Service to track and manage detection history to avoid sending duplicate frames.
 * Uses detectedPeopleKey per camera to identify duplicate person detections.
 * Also performs frame similarity analysis to prevent duplicate notifications when the same
 * person is detected with different confidence scores (known -> unknown).
 */
@Log4j2
@Service
public class DetectionHistoryService {

    // 5 seconds cooldown per person per camera
    private static final long DETECTION_COOLDOWN_MS = 5 * 1000; // 5 seconds

    // Time window to check for similar frames (3 seconds)
    private static final long SIMILARITY_CHECK_WINDOW_MS = 5 * 1000;

    // Similarity threshold (0-100, where lower is more similar)
    // If hash difference is below this threshold, frames are considered similar
    private static final int SIMILARITY_THRESHOLD = 15;

    private final Map<String, DetectionRecord> detectionHistory = new ConcurrentHashMap<>();

    // Track recent detections per camera to check for similar frames
    private final Map<String, CameraDetectionHistory> cameraHistory = new ConcurrentHashMap<>();

    // Queue for pending unknown detections that are waiting to see if they get recognized
    private final Map<String, PendingUnknownDetection> pendingUnknownDetections = new ConcurrentHashMap<>();

    // Time to wait before sending unknown notification (3 seconds)
    private static final long UNKNOWN_WAIT_TIME_MS = 3 * 1000;

    /**
     * Checks if a KNOWN person detection should be sent
     * Also cancels any pending unknown notifications for similar faces
     *
     * @param imageHash            Hash of the image content
     * @param detectedPeopleKey    Unique identifier for the detected person
     * @param cameraName           Name of the camera
     * @param faceHash             Hash of the extracted face region
     * @return true if the detection should be sent, false if it's a duplicate
     */
    public boolean shouldSendDetection(String imageHash, String detectedPeopleKey, String cameraName, byte[] faceHash) {
        String compositeKey = cameraName + ":" + detectedPeopleKey;
        long now = System.currentTimeMillis();

        // Cancel any pending unknown notifications for similar faces
        cancelSimilarPendingUnknownDetections(cameraName, faceHash, now);

        DetectionRecord lastDetection = detectionHistory.get(compositeKey);

        if (lastDetection == null) {
            // First detection of this person for this camera
            detectionHistory.put(compositeKey, new DetectionRecord(imageHash, now, detectedPeopleKey, faceHash));

            // Update camera history
            updateCameraHistory(cameraName, imageHash, now, detectedPeopleKey, faceHash);

            log.debug("New person detection recorded for {}: {}", cameraName, detectedPeopleKey);
            return true;
        }

        long timeSinceLastDetection = now - lastDetection.timestamp;

        if (timeSinceLastDetection < DETECTION_COOLDOWN_MS) {
            log.debug("Duplicate person detection filtered for {}:{} ({}ms since last)", cameraName, detectedPeopleKey, timeSinceLastDetection);
            return false;
        }

        // Cooldown expired, update record and allow
        detectionHistory.put(compositeKey, new DetectionRecord(imageHash, now, detectedPeopleKey, faceHash));

        // Update camera history
        updateCameraHistory(cameraName, imageHash, now, detectedPeopleKey, faceHash);

        log.debug("Detection cooldown expired for {}:{}, allowing new notification ({}ms since last)", cameraName, detectedPeopleKey, timeSinceLastDetection);
        return true;
    }

    /**
     * Update camera-specific detection history
     */
    private void updateCameraHistory(String cameraName, String imageHash, long timestamp, String detectedPeopleKey, byte[] faceHash) {
        CameraDetectionHistory camHistory = cameraHistory.computeIfAbsent(cameraName, k -> new CameraDetectionHistory());
        boolean isUnknown = detectedPeopleKey.contains("Unknown");
        camHistory.addDetection(imageHash, timestamp, isUnknown, faceHash);
    }

    /**
     * Cancel any pending unknown notifications for similar faces when a known person is detected
     */
    private void cancelSimilarPendingUnknownDetections(String cameraName, byte[] faceHash, long currentTime) {
        if (faceHash == null) {
            return; // Can't compare without face hash
        }

        pendingUnknownDetections.entrySet().removeIf(entry -> {
            PendingUnknownDetection pending = entry.getValue();
            if (!pending.cameraName.equals(cameraName)) {
                return false; // Different camera
            }

            if (pending.faceHash == null) {
                return false; // Can't compare
            }

            int similarity = computeFaceHashSimilarity(faceHash, pending.faceHash);
            if (similarity <= SIMILARITY_THRESHOLD) {
                log.info("Cancelling pending Unknown notification for camera '{}' - known person now detected with similar face (similarity: {})",
                        cameraName, similarity);
                return true; // Remove this pending detection
            }
            return false;
        });
    }

    /**
     * Queue an unknown detection for delayed notification
     * Returns the pending key if queued, null if it should be skipped
     */
    public String queueUnknownDetection(String imageHash, String detectedPeopleKey, String cameraName,
                                       byte[] imageBytes, Map<String, Double> detectedPeopleWithScores, byte[] faceHash) {
        long now = System.currentTimeMillis();

        // Check if we recently detected a known person with similar face
        CameraDetectionHistory camHistory = cameraHistory.get(cameraName);
        if (camHistory != null) {
            DetectionRecord recentKnown = camHistory.getRecentKnownPerson(now);
            if (recentKnown != null && recentKnown.faceHash != null && faceHash != null) {
                int similarity = computeFaceHashSimilarity(faceHash, recentKnown.faceHash);
                if (similarity <= SIMILARITY_THRESHOLD) {
                    log.info("Blocking Unknown notification for camera '{}' - similar face with known person detected {}ms ago (similarity: {})",
                            cameraName, now - recentKnown.timestamp, similarity);
                    return null;
                }
            }
        }

        // Check if there's already a pending unknown detection with similar face
        String pendingKey = findSimilarPendingUnknown(cameraName, faceHash, now);
        if (pendingKey != null) {
            log.debug("Similar unknown face already pending for camera '{}', skipping duplicate", cameraName);
            return null;
        }

        // Add to pending queue - wait 3 seconds before actually sending notification
        String pendingUnknownKey = cameraName + ":" + System.nanoTime(); // Unique key
        pendingUnknownDetections.put(pendingUnknownKey,
                new PendingUnknownDetection(imageHash, now, cameraName, detectedPeopleKey, imageBytes, detectedPeopleWithScores, faceHash));
        log.info("Unknown detection queued for camera '{}' - waiting {}ms to check if person gets recognized", cameraName, UNKNOWN_WAIT_TIME_MS);

        return pendingUnknownKey;
    }

    /**
     * Get pending unknown detections that are ready to be sent (waited 3 seconds)
     * This should be called periodically to process the queue
     */
    public Map<String, PendingUnknownDetection> getReadyUnknownDetections() {
        long now = System.currentTimeMillis();
        Map<String, PendingUnknownDetection> readyDetections = new ConcurrentHashMap<>();

        pendingUnknownDetections.entrySet().removeIf(entry -> {
            PendingUnknownDetection pending = entry.getValue();
            long waitTime = now - pending.timestamp;

            if (waitTime >= UNKNOWN_WAIT_TIME_MS) {
                // This detection has waited long enough
                log.info("Unknown detection for camera '{}' is ready to send after waiting {}ms",
                        pending.cameraName, waitTime);
                readyDetections.put(entry.getKey(), pending);
                return true; // Remove from pending queue
            }
            return false; // Keep waiting
        });

        return readyDetections;
    }

    /**
     * Find if there's already a similar pending unknown detection for this camera
     */
    private String findSimilarPendingUnknown(String cameraName, byte[] faceHash, long currentTime) {
        if (faceHash == null) {
            return null;
        }

        for (Map.Entry<String, PendingUnknownDetection> entry : pendingUnknownDetections.entrySet()) {
            PendingUnknownDetection pending = entry.getValue();
            if (pending.cameraName.equals(cameraName) && pending.faceHash != null) {
                int similarity = computeFaceHashSimilarity(faceHash, pending.faceHash);
                if (similarity <= SIMILARITY_THRESHOLD) {
                    return entry.getKey();
                }
            }
        }
        return null;
    }

    /**
     * Compute similarity between two face hashes using byte array comparison
     * Returns a score from 0 (identical) to 100 (completely different)
     */
    private int computeFaceHashSimilarity(byte[] hash1, byte[] hash2) {
        if (hash1 == null || hash2 == null) {
            return 100; // Completely different
        }

        // If lengths are very different, they're not similar
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

        // Add difference in length
        differentBytes += Math.abs(hash1.length - hash2.length);

        // Convert to percentage (0-100)
        return Math.min(100, (differentBytes * 100) / Math.max(hash1.length, hash2.length));
    }

    /**
     * Compute similarity between two image hashes
     * Returns a score from 0 (identical) to 100 (completely different)
     */
    private int computeHashSimilarity(String hash1, String hash2) {
        if (hash1 == null || hash2 == null || hash1.length() != hash2.length()) {
            return 100; // Completely different
        }

        int differentChars = 0;
        int totalChars = hash1.length();

        for (int i = 0; i < totalChars; i++) {
            if (hash1.charAt(i) != hash2.charAt(i)) {
                differentChars++;
            }
        }

        // Convert to percentage (0-100)
        return (differentChars * 100) / totalChars;
    }

    /**
     * Computes SHA-256 hash of image bytes
     *
     * @param imageBytes The image data
     * @return Hex string representation of the hash
     */
    public String computeImageHash(byte[] imageBytes) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hash = digest.digest(imageBytes);
            StringBuilder hexString = new StringBuilder();
            for (byte b : hash) {
                String hex = Integer.toHexString(0xff & b);
                if (hex.length() == 1) {
                    hexString.append('0');
                }
                hexString.append(hex);
            }
            return hexString.toString();
        } catch (Exception e) {
            log.error("Error computing image hash", e);
            return String.valueOf(System.identityHashCode(imageBytes));
        }
    }

    /**
     * Clears old detection records to prevent memory leaks
     * Called periodically to clean up history older than 10 minutes
     */
    public void cleanupOldRecords() {
        long tenMinutesAgo = System.currentTimeMillis() - (10 * 60 * 1000);
        detectionHistory.entrySet().removeIf(entry -> entry.getValue().timestamp < tenMinutesAgo);

        // Cleanup camera history
        cameraHistory.values().forEach(history -> history.cleanup(tenMinutesAgo));

        // Cleanup stale pending unknown detections (older than 1 minute)
        long oneMinuteAgo = System.currentTimeMillis() - (60 * 1000);
        pendingUnknownDetections.entrySet().removeIf(entry -> entry.getValue().timestamp < oneMinuteAgo);

        log.debug("Cleaned up old detection records");
    }

    /**
     * Internal class to store detection information
     */
    private static class DetectionRecord {
        String imageHash;
        long timestamp;
        String personKey;
        byte[] faceHash; // Hash of extracted face region

        DetectionRecord(String imageHash, long timestamp, String personKey, byte[] faceHash) {
            this.imageHash = imageHash;
            this.timestamp = timestamp;
            this.personKey = personKey;
            this.faceHash = faceHash;
        }

        DetectionRecord(String imageHash, long timestamp, String personKey) {
            this(imageHash, timestamp, personKey, null);
        }
    }

    /**
     * Stores information about a pending unknown detection
     */
    public static class PendingUnknownDetection {
        public final String imageHash;
        public final long timestamp;
        public final String cameraName;
        public final String detectedPeopleKey;
        public final byte[] imageBytes;
        public final Map<String, Double> detectedPeopleWithScores;
        public final byte[] faceHash; // Hash of the extracted face region for comparison

        public PendingUnknownDetection(String imageHash, long timestamp, String cameraName, String detectedPeopleKey,
                                      byte[] imageBytes, Map<String, Double> detectedPeopleWithScores, byte[] faceHash) {
            this.imageHash = imageHash;
            this.timestamp = timestamp;
            this.cameraName = cameraName;
            this.detectedPeopleKey = detectedPeopleKey;
            this.imageBytes = imageBytes;
            this.detectedPeopleWithScores = detectedPeopleWithScores;
            this.faceHash = faceHash;
        }
    }

    /**
     * Tracks recent detections per camera to enable similarity checking
     */
    private static class CameraDetectionHistory {
        private DetectionRecord lastKnownPerson;
        private DetectionRecord lastUnknownPerson;

        /**
         * Add a detection to camera history
         */
        void addDetection(String imageHash, long timestamp, boolean isUnknown, byte[] faceHash) {
            DetectionRecord record = new DetectionRecord(imageHash, timestamp, null, faceHash);
            if (isUnknown) {
                lastUnknownPerson = record;
            } else {
                lastKnownPerson = record;
            }
        }

        /**
         * Get the most recent known person detection within the similarity check window
         */
        DetectionRecord getRecentKnownPerson(long currentTime) {
            if (lastKnownPerson != null) {
                long timeSince = currentTime - lastKnownPerson.timestamp;
                if (timeSince <= SIMILARITY_CHECK_WINDOW_MS) {
                    return lastKnownPerson;
                }
            }
            return null;
        }

        /**
         * Cleanup old records
         */
        void cleanup(long cutoffTime) {
            if (lastKnownPerson != null && lastKnownPerson.timestamp < cutoffTime) {
                lastKnownPerson = null;
            }
            if (lastUnknownPerson != null && lastUnknownPerson.timestamp < cutoffTime) {
                lastUnknownPerson = null;
            }
        }
    }
}
