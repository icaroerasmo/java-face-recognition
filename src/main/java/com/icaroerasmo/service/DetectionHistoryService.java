package com.icaroerasmo.service;

import lombok.extern.log4j.Log4j2;
import org.springframework.stereotype.Service;

import java.security.MessageDigest;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Service to track and manage detection history to avoid sending duplicate frames.
 * Uses detectedPeopleKey per camera to identify duplicate person detections.
 */
@Log4j2
@Service
public class DetectionHistoryService {

    // 5 seconds cooldown per person per camera
    private static final long DETECTION_COOLDOWN_MS = 5 * 1000; // 5 seconds
    private final Map<String, DetectionRecord> detectionHistory = new ConcurrentHashMap<>();

    /**
     * Checks if a detection should be sent or if it's a duplicate
     * Duplicate is determined by the same detectedPeopleKey in the same camera within the cooldown.
     *
     * @param imageHash            Hash of the image content
     * @param detectedPeopleKey    Unique identifier for the detected person (e.g. face id or embedding key)
     * @param cameraName           Name of the camera
     * @return true if the detection is new or cooldown has expired, false if it's a duplicate
     */
    public boolean shouldSendDetection(String imageHash, String detectedPeopleKey, String cameraName) {
        String compositeKey = cameraName + ":" + detectedPeopleKey;
        long now = System.currentTimeMillis();

        DetectionRecord lastDetection = detectionHistory.get(compositeKey);

        if (lastDetection == null) {
            // First detection of this person for this camera
            detectionHistory.put(compositeKey, new DetectionRecord(imageHash, now));
            log.debug("New person detection recorded for {}: {}", cameraName, detectedPeopleKey);
            return true;
        }

        long timeSinceLastDetection = now - lastDetection.timestamp;

        if (timeSinceLastDetection < DETECTION_COOLDOWN_MS) {
            log.debug("Duplicate person detection filtered for {}:{} ({}ms since last)", cameraName, detectedPeopleKey, timeSinceLastDetection);
            return false;
        }

        // Cooldown expired, update record and allow
        detectionHistory.put(compositeKey, new DetectionRecord(imageHash, now));
        log.debug("Detection cooldown expired for {}:{}, allowing new notification ({}ms since last)", cameraName, detectedPeopleKey, timeSinceLastDetection);
        return true;
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
        log.debug("Cleaned up old detection records");
    }

    /**
     * Internal class to store detection information
     */
    private static class DetectionRecord {
        String imageHash;
        long timestamp;

        DetectionRecord(String imageHash, long timestamp) {
            this.imageHash = imageHash;
            this.timestamp = timestamp;
        }
    }
}
