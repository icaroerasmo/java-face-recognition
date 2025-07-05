package com.icaroerasmo.service;

import com.icaroerasmo.model.DetectionRecord;
import lombok.extern.log4j.Log4j2;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

@Log4j2
@Service
public class DetectionService {
    private static final int MIN_DETECTION_WINDOW_SECONDS = 3;
    private static final int MIN_DETECTIONS = 5;
    private static final String UNKNOWN = "Unknown";

    private final Map<String, List<DetectionRecord>> detectionBuffer = new ConcurrentHashMap<>();
    private final Map<String, Boolean> announceFlags = new ConcurrentHashMap<>();

    public boolean shouldAnnounceDetection(String personName, Double score) {
        if (personName == null || score == null) {
            return false;
        }

        Instant now = Instant.now();
        detectionBuffer.computeIfAbsent(personName, k -> Collections.synchronizedList(new ArrayList<>()))
                .add(new DetectionRecord(personName, score, now));

        return announceFlags.getOrDefault(personName, false);
    }

    @Scheduled(fixedDelay = 1000)
    private void checkDetections() {
        Instant now = Instant.now();
        boolean hasOtherDetections = detectionBuffer.keySet().stream()
                .anyMatch(name -> !UNKNOWN.equals(name));

        detectionBuffer.forEach((personName, detections) -> {
            cleanOldDetections(personName, now);
            Instant windowStart = now.minusSeconds(MIN_DETECTION_WINDOW_SECONDS);
            long recentDetections = detections.stream()
                    .filter(d -> !d.getTimestamp().isBefore(windowStart))
                    .count();

            if (recentDetections < MIN_DETECTIONS) {
                if (!UNKNOWN.equals(personName) || !hasOtherDetections) {
                    announceFlags.put(personName, true);
                }
                detections.clear();
            } else {
                announceFlags.put(personName, false);
            }
        });
    }

    private void cleanOldDetections(String personName, Instant now) {
        List<DetectionRecord> records = detectionBuffer.get(personName);
        if (records != null) {
            records.removeIf(record ->
                    Duration.between(record.getTimestamp(), now).getSeconds() > MIN_DETECTION_WINDOW_SECONDS);
        }
    }
}
