package com.icaroerasmo.detectors.person.services;

import lombok.extern.log4j.Log4j2;
import org.springframework.stereotype.Service;

import java.security.MessageDigest;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import static com.icaroerasmo.utils.FaceHashUtils.computeSimilarity;

/**
 * Service to track and manage detection history to avoid sending duplicate frames.
 * Uses detectedPeopleKey per camera to identify duplicate person detections.
 */
@Log4j2
@Service
public class DetectionHistoryService {

    // 5 seconds cooldown per person per camera
    private static final long DETECTION_COOLDOWN_MS = 5 * 1000;

    // Similarity threshold (0-100, where lower is more similar)
    private static final int SIMILARITY_THRESHOLD = 15;

    private final Map<String, DetectionRecord> detectionHistory = new ConcurrentHashMap<>();

    // Track recent detections per camera to check for similar frames
    private final Map<String, CameraDetectionHistory> cameraHistory = new ConcurrentHashMap<>();

    /**
     * Checks if an UNKNOWN person detection should be sent (after tracking)
     * This is similar to shouldSendDetection but without canceling pending unknowns
     * since tracking replaces the pending queue system
     *
     * @param imageHash            Hash of the image content
     * @param detectedPeopleKey    Unique identifier for the detected person
     * @param cameraName           Name of the camera
     * @param faceHash             Hash of the extracted face region
     * @return true if the detection should be sent, false if it's a duplicate
     */
    public boolean shouldSendUnknownDetection(String imageHash, String detectedPeopleKey, String cameraName, byte[] faceHash) {
        String compositeKey = cameraName + ":" + detectedPeopleKey;
        long now = System.currentTimeMillis();

        DetectionRecord lastDetection = detectionHistory.get(compositeKey);

        if (lastDetection == null) {
            // First detection of this unknown person for this camera
            // Don't add to history yet - will be added after successful send
            log.debug("New unknown person detection for {}: {}", cameraName, detectedPeopleKey);
            return true;
        }

        long timeSinceLastDetection = now - lastDetection.timestamp;

        if (timeSinceLastDetection < DETECTION_COOLDOWN_MS) {
            // Check face similarity if available
            if (lastDetection.faceHash != null && faceHash != null) {
                int similarity = computeSimilarity(faceHash, lastDetection.faceHash);
                if (similarity <= SIMILARITY_THRESHOLD) {
                    log.debug("Duplicate unknown person filtered for {}:{} ({}ms since last, similarity: {})",
                            cameraName, detectedPeopleKey, timeSinceLastDetection, similarity);
                    return false;
                }
            } else {
                log.debug("Duplicate unknown person filtered for {}:{} ({}ms since last)",
                        cameraName, detectedPeopleKey, timeSinceLastDetection);
                return false;
            }
        }

        // Cooldown expired or face is different enough
        log.debug("Unknown detection cooldown expired for {}:{}, allowing new notification ({}ms since last)",
                cameraName, detectedPeopleKey, timeSinceLastDetection);
        return true;
    }

    /**
     * Checks if a person detection should be sent
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

        DetectionRecord lastDetection = detectionHistory.get(compositeKey);

        if (lastDetection == null) {
            // First detection of this person for this camera
            detectionHistory.put(compositeKey, new DetectionRecord(imageHash, now, detectedPeopleKey, faceHash));
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
     * Mark an unknown detection as sent to prevent duplicate notifications
     */
    public void markUnknownDetectionAsSent(String imageHash, String detectedPeopleKey, String cameraName, byte[] faceHash) {
        String compositeKey = cameraName + ":" + detectedPeopleKey;
        long now = System.currentTimeMillis();
        detectionHistory.put(compositeKey, new DetectionRecord(imageHash, now, detectedPeopleKey, faceHash));
        updateCameraHistory(cameraName, imageHash, now, detectedPeopleKey, faceHash);
        log.debug("Marked unknown detection as sent for {}: {}", cameraName, detectedPeopleKey);
    }


    /**
     * Computes SHA-256 hash of image bytes
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
     */
    public void cleanupOldRecords() {
        long tenMinutesAgo = System.currentTimeMillis() - (10 * 60 * 1000);
        detectionHistory.entrySet().removeIf(entry -> entry.getValue().timestamp < tenMinutesAgo);
        cameraHistory.values().forEach(history -> history.cleanup(tenMinutesAgo));

        log.debug("Cleaned up old detection records");
    }

    /**
     * Internal class to store detection information
     */
    private static class DetectionRecord {
        String imageHash;
        long timestamp;
        String personKey;
        byte[] faceHash;

        DetectionRecord(String imageHash, long timestamp, String personKey, byte[] faceHash) {
            this.imageHash = imageHash;
            this.timestamp = timestamp;
            this.personKey = personKey;
            this.faceHash = faceHash;
        }
    }

    /**
     * Tracks recent detections per camera
     */
    private static class CameraDetectionHistory {
        private DetectionRecord lastKnownPerson;
        private DetectionRecord lastUnknownPerson;

        void addDetection(String imageHash, long timestamp, boolean isUnknown, byte[] faceHash) {
            DetectionRecord record = new DetectionRecord(imageHash, timestamp, null, faceHash);
            if (isUnknown) {
                lastUnknownPerson = record;
            } else {
                lastKnownPerson = record;
            }
        }

        DetectionRecord getRecentKnownPerson(long currentTime) {
            if (lastKnownPerson != null) {
                long timeSince = currentTime - lastKnownPerson.timestamp;
                if (timeSince <= 5000) { // 5 seconds
                    return lastKnownPerson;
                }
            }
            return null;
        }

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
