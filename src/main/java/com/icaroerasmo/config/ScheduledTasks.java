package com.icaroerasmo.config;

import com.icaroerasmo.properties.TrainingProperties;
import com.icaroerasmo.service.DetectionHistoryService;
import com.icaroerasmo.service.FaceRecognitionService;
import com.icaroerasmo.service.FaceRecognizerHolder;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.bytedeco.opencv.opencv_face.FaceRecognizer;
import org.springframework.scheduling.annotation.EnableScheduling;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.io.File;
import java.nio.file.Path;

/**
 * Scheduled tasks for application maintenance
 */
@Log4j2
@Component
@EnableScheduling
@RequiredArgsConstructor
public class ScheduledTasks {

    private final DetectionHistoryService detectionHistoryService;
    private final FaceRecognitionService faceRecognitionService;
    private final FaceRecognizerHolder faceRecognizerHolder;
    private final TrainingProperties trainingProperties;

    /**
     * Cleanup old detection records every 10 minutes
     * This prevents memory leaks from accumulating detection history
     */
    @Scheduled(fixedRate = 10 * 60 * 1000) // 10 minutes
    public void cleanupDetectionHistory() {
        log.debug("Running scheduled cleanup of detection history");
        detectionHistoryService.cleanupOldRecords();
    }

    /**
     * Check for training data changes every minute
     * If changes are detected, retrain the model and update the database
     */
    @Scheduled(fixedRate = 60 * 1000 * 10) // 1 minute
    public void checkTrainingDataChanges() {
        try {
            Path trainingRootPath = getTrainingRootPath();

            if (trainingRootPath == null || !trainingRootPath.toFile().exists()) {
                log.trace("Training folder not found, skipping scheduled check");
                return;
            }

            log.trace("Checking for training data changes in: {}", trainingRootPath);

            boolean hasChanges = faceRecognitionService.isTrainingDataChanged(trainingRootPath);

            if (hasChanges) {
                log.warn("=== TRAINING DATA CHANGES DETECTED ===");
                log.info("Starting automatic retraining due to detected changes...");

                FaceRecognizer newRecognizer = faceRecognitionService.train(trainingRootPath);
                faceRecognizerHolder.updateRecognizer(newRecognizer);

                log.info("=== AUTOMATIC RETRAINING COMPLETED SUCCESSFULLY ===");
                log.info("Model updated in database and active recognizer replaced");
            } else {
                log.trace("No training data changes detected");
            }
        } catch (Exception e) {
            log.error("Error during scheduled training data check", e);
        }
    }

    /**
     * Get the training root path from configuration
     */
    private Path getTrainingRootPath() {
        String trainingRootFolder = trainingProperties.getRootFolder();

        // Try filesystem path
        File filesystemFolder = new File(trainingRootFolder);
        if (filesystemFolder.exists() && filesystemFolder.isDirectory()) {
            return filesystemFolder.toPath();
        }

        // Try Docker path
        File dockerFolder = new File("/app/train");
        if (dockerFolder.exists() && dockerFolder.isDirectory()) {
            return dockerFolder.toPath();
        }

        // Path not found
        return null;
    }
}
