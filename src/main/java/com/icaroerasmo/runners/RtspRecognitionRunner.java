package com.icaroerasmo.runners;

import com.icaroerasmo.detectors.movement.MovementAlertPolicy;
import com.icaroerasmo.detectors.movement.MovementDetector;
import com.icaroerasmo.detectors.movement.MovementResultStore;
import com.icaroerasmo.detectors.person.services.FaceRecognitionRuntime;
import com.icaroerasmo.detectors.person.services.FaceRecognitionService;
import com.icaroerasmo.detectors.person.services.FaceRecognizerHolderService;
import com.icaroerasmo.pipeline.CameraPipeline;
import com.icaroerasmo.pipeline.RtspCameraStreamWorker;
import com.icaroerasmo.properties.CameraProperties;
import com.icaroerasmo.properties.StreamsProperties;
import com.icaroerasmo.properties.TrainingProperties;
import com.icaroerasmo.service.RtspFrameExtractorService;
import com.icaroerasmo.service.TelegramPublisherService;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.jetbrains.annotations.NotNull;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Component;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

@Log4j2
@Component
@RequiredArgsConstructor
public class RtspRecognitionRunner {

    private final FaceRecognitionService faceRecognitionService;
    private final FaceRecognizerHolderService faceRecognizerHolderService;
    private final RtspFrameExtractorService rtspFrameExtractorService;
    private final TelegramPublisherService telegramPublisherService;
    private final StreamsProperties streamsProperties;
    private final TrainingProperties trainingProperties;
    private final CameraPipeline cameraPipeline;
    private final MovementDetector movementDetector;
    private final MovementResultStore movementResultStore;
    @Qualifier("movementAlertPolicy")
    private final MovementAlertPolicy movementAlertPolicy;
    @Qualifier("petAlertPolicy")
    private final MovementAlertPolicy petAlertPolicy;

    /**
     * Startup orchestration only. The per-camera reconnect/hibernate/backoff
     * lifecycle and the per-frame pipeline live in {@link RtspCameraStreamWorker}
     * and {@link CameraPipeline}.
     */
    public void start(String... args) throws Exception {

        try {
            File trainingRootDir = getTrainedFile();
            FaceRecognitionRuntime faceRecognizer = faceRecognitionService.ensureTrained(trainingRootDir.toPath());

            // Initialize the holder with the trained recognizer
            faceRecognizerHolderService.updateRecognizer(faceRecognizer);

            List<CameraProperties> cameraProperties = streamsProperties.getCameras();
            if (cameraProperties == null || cameraProperties.isEmpty()) {
                throw new IllegalStateException("No cameras configured under object-detection.streams.cameras");
            }

            // Create thread pool with one thread per camera
            ExecutorService executorService = Executors.newFixedThreadPool(cameraProperties.size());
            List<Future<?>> futures = new ArrayList<>();

            for (CameraProperties camera : cameraProperties) {
                if (camera == null || camera.getUrl() == null || camera.getUrl().isBlank()) {
                    continue;
                }

                // Submit camera processing task to executor
                Future<?> future = executorService.submit(new RtspCameraStreamWorker(
                    camera,
                    rtspFrameExtractorService,
                    telegramPublisherService,
                    cameraPipeline,
                    movementDetector,
                    movementResultStore,
                    movementAlertPolicy,
                    petAlertPolicy
                ));
                futures.add(future);
            }

            // Wait for all camera streams to complete (they run indefinitely, so this won't return)
            for (Future<?> future : futures) {
                try {
                    future.get(); // This will block if the task is still running
                } catch (Exception e) {
                    log.error("Camera stream processing failed", e);
                }
            }

            executorService.shutdown();

        } catch (Exception e) {
            log.error("Error in RtspRecognitionRunner", e);
            throw e;
        }
    }

    @NotNull
    private File getTrainedFile() {
        String trainingRootFolder = trainingProperties.getRootFolder();

        // Try 1: Check if it's an absolute path or relative path on filesystem
        File filesystemFolder = new File(trainingRootFolder);
        if (filesystemFolder.exists() && filesystemFolder.isDirectory()) {
            log.debug("Found training folder on filesystem: {}", filesystemFolder.getAbsolutePath());
            return filesystemFolder;
        }

        // Try 2: Check in /app/train (Docker runtime path)
        File dockerFolder = new File("/app/train");
        if (dockerFolder.exists() && dockerFolder.isDirectory()) {
            log.debug("Found training folder in Docker path: {}", dockerFolder.getAbsolutePath());
            return dockerFolder;
        }

        // Try 3: Check classpath (development/JAR with embedded resources)
        ClassPathResource trainingResource = new ClassPathResource(trainingRootFolder);
        if (trainingResource.exists()) {
            try {
                File trainingRootDir = trainingResource.getFile();
                log.debug("Found training folder on classpath: {}", trainingRootDir.getAbsolutePath());
                return trainingRootDir;
            } catch (IOException ex) {
                log.warn("Unable to extract training folder from JAR classpath, will try to use JAR resources", ex);
                // If JAR extraction fails, we'll create a fallback or error
                throw new IllegalStateException("Training root folder '" + trainingRootFolder + "' found in JAR but cannot be extracted. " +
                        "For Docker, mount training data at /app/train or set TRAINING_ROOT_FOLDER environment variable.", ex);
            }
        }

        // No training folder found anywhere
        throw new IllegalStateException(
                "Training root folder '" + trainingRootFolder + "' not found in any of the following locations:\n" +
                "1. Filesystem path: " + filesystemFolder.getAbsolutePath() + "\n" +
                "2. Docker path: /app/train\n" +
                "3. Classpath resource: " + trainingRootFolder + "\n" +
                "Please ensure your training dataset is available in one of these locations.");
    }

}
