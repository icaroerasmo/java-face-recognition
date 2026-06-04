package com.icaroerasmo.service;

import com.icaroerasmo.model.FaceRecognition;
import com.icaroerasmo.utils.MatUtil;
import com.icaroerasmo.properties.TrainingProperties;
import com.icaroerasmo.repository.TrainingMetadataRepository;
import com.icaroerasmo.repository.TrainedDatasetRepository;
import com.icaroerasmo.repository.entity.TrainingMetadata;
import com.icaroerasmo.repository.entity.TrainedDataset;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.bytedeco.opencv.opencv_core.*;
import org.springframework.stereotype.Service;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.opencv.opencv_face.FaceRecognizer;
import org.bytedeco.opencv.opencv_face.LBPHFaceRecognizer;
import org.springframework.transaction.annotation.Transactional;

import java.io.File;
import java.io.IOException;
import java.nio.IntBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.stream.Collectors.toMap;
import static org.bytedeco.opencv.global.opencv_core.CV_32SC1;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;

@Log4j2
@Service
@RequiredArgsConstructor
public class FaceRecognitionService {

    private final DeepLearningFaceDetectionService deepLearningFaceDetectionService;
    private final MatUtil matUtil;
    private final TrainingProperties trainingProperties;
    private final TrainingMetadataRepository trainingMetadataRepository;
    private final TrainedDatasetRepository trainedDatasetRepository;

    // Base distance threshold for reference (medium-sized faces at typical resolution)
    public static final double BASE_DISTANCE = 60.0;
    public static final String UNKNOWN = "Unknown";

    // Expected face ratio for calibration (5% of frame area is typical for good recognition)
    private static final double EXPECTED_FACE_RATIO = 0.05;

    // Min and max adaptive thresholds
    private static final double MIN_ADAPTIVE_THRESHOLD = 30.0;
    private static final double MAX_ADAPTIVE_THRESHOLD = 90.0;

    public FaceRecognizer load() {
        FaceRecognizer faceRecognizer = LBPHFaceRecognizer.create();
        TrainedDataset trained = trainedDatasetRepository.findAll().stream().findFirst()
                .orElseThrow(() -> new IllegalStateException("No trained dataset found in database"));
        Path tmp = null;
        try {
            tmp = Files.createTempFile("trained_dataset", ".xml");
            Files.write(tmp, trained.getModelXml());
            faceRecognizer.read(tmp.toString());
        } catch (IOException e) {
            throw new IllegalStateException("Failed to load trained dataset from database", e);
        } finally {
            if (tmp != null) {
                try {
                    Files.deleteIfExists(tmp);
                } catch (IOException ignore) {
                }
            }
        }
        return faceRecognizer;
    }

    /**
     * Test face recognition on an image.
     * The FaceRecognizer.predict() operation is synchronized per recognizer instance
     * to prevent race conditions when multiple cameras are processing frames simultaneously.
     */
    public FaceRecognition test(FaceRecognizer faceRecognizer, Mat testImage) {

        // Validate input
        if (testImage == null || testImage.empty()) {
            log.warn("Cannot perform face recognition on null or empty image");
            return new FaceRecognition(List.of(), testImage);
        }

        // Calculate frame area for adaptive threshold calculation
        final double frameArea = testImage.rows() * testImage.cols();
        log.debug("Frame resolution: {}x{}, total area: {} pixels",
            testImage.cols(), testImage.rows(), (long)frameArea);

        List<FaceRecognition.DetectedFaces> detectedFaces = deepLearningFaceDetectionService.detect(testImage).stream().map(faceRect -> {

            Mat extractedFace = null;
            Mat img = null;

            try {
                // Extract the face region
                extractedFace = new Mat(testImage, faceRect);

                // Validate that the extracted face is not empty
                if (extractedFace.empty()) {
                    log.warn("Extracted face Mat is empty for rect: x={}, y={}, width={}, height={}",
                            faceRect.x(), faceRect.y(), faceRect.width(), faceRect.height());
                    return null;
                }

                // Validate dimensions
                if (extractedFace.rows() <= 0 || extractedFace.cols() <= 0) {
                    log.warn("Extracted face has invalid dimensions: rows={}, cols={}",
                            extractedFace.rows(), extractedFace.cols());
                    return null;
                }

                img = matUtil.convertToGray(extractedFace);

                if (img.empty()) {
                    log.warn("Converted gray image is empty");
                    return null;
                }

                IntPointer detectedPersonPtr = new IntPointer(1);
                DoublePointer distancePtr = new DoublePointer(1);

                // CRITICAL: Synchronize on the FaceRecognizer instance to prevent concurrent access
                // The recognizer maintains internal state that can be corrupted by concurrent predictions
                synchronized (faceRecognizer) {
                    faceRecognizer.predict(img, detectedPersonPtr, distancePtr);

                    String label = faceRecognizer.getLabelInfo(detectedPersonPtr.get(0)).getString();
                    String detectedPerson = sanitizeLabel(label);
                    double detectionDistance = distancePtr.get(0);

                    // Calculate adaptive threshold based on face size relative to frame
                    double faceArea = faceRect.width() * faceRect.height();
                    double faceRatio = faceArea / frameArea;
                    double adaptiveThreshold = calculateAdaptiveThreshold(faceRatio);

                    if (detectionDistance > adaptiveThreshold) {
                        log.debug("Detected person is {} with distance {} (adaptive threshold: {}, face ratio: {}%) - classified as {}",
                                detectedPerson,
                                String.format("%.2f", detectionDistance),
                                String.format("%.2f", adaptiveThreshold),
                                String.format("%.4f", faceRatio * 100),
                                UNKNOWN);
                        detectedPerson = UNKNOWN;
                    } else {
                        log.debug("Detected person is {} with distance {} (adaptive threshold: {}, face ratio: {}%)",
                                detectedPerson,
                                String.format("%.2f", detectionDistance),
                                String.format("%.2f", adaptiveThreshold),
                                String.format("%.4f", faceRatio * 100));
                    }

                    return new FaceRecognition.DetectedFaces(detectedPerson, detectionDistance, faceRect);
                }
            } catch(Exception e) {
                log.error("Error processing face detection for rect: x={}, y={}, width={}, height={}",
                        faceRect.x(), faceRect.y(), faceRect.width(), faceRect.height(), e);
                return null;
            } finally {
                matUtil.releaseResources(extractedFace, img);
            }
        }).filter(detected -> detected != null).toList();

        return new FaceRecognition(detectedFaces, testImage);
    }

    @Transactional
    public FaceRecognizer train(Path rootFolder) throws IOException {
        Path rootFolderPath = rootFolder;

        // Use try-with-resources to properly close streams
        Map<Path, Object[]> fileList;
        try (var rootStream = Files.list(rootFolderPath)) {
            fileList = rootStream
                .filter(file -> file.toFile().isDirectory())
                .flatMap(folder -> {
                    try {
                        var personName = folder.getName(folder.getNameCount() - 1).toString();
                        // Need to collect to list to avoid nested stream issues
                        return Files.list(folder).map(file -> Map.entry(file, personName));
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                })
                .map(entry -> {
                    File image = entry.getKey().toFile();

                    Mat img = imread(image.getAbsolutePath());

                    List<Rect> facesList = deepLearningFaceDetectionService.detect(img);

                    if (facesList.isEmpty()) {
                        return null;
                    }

                    Rect faceRect = facesList.getFirst();
                    Mat face = new Mat(img, faceRect);

                    matUtil.releaseResources(img);

                    return Map.entry(entry.getKey(), new Object[]{face, entry.getValue()});
                })
                .filter(entry -> entry != null)
                .collect(toMap(Map.Entry::getKey, Map.Entry::getValue));
        }

        MatVector images = new MatVector(fileList.size());

        List<String> strLabels = fileList.values().stream().map(data -> (String)data[1]).distinct().toList();

        Mat labels = new Mat(fileList.size(), 1, CV_32SC1);
        IntBuffer labelsBuf = labels.createBuffer();

        final AtomicInteger counter = new AtomicInteger();

        fileList.keySet().forEach(path -> {

            Object[] data = fileList.get(path);

            Mat img = matUtil.convertToGray((Mat)data[0]);

            images.put(counter.get(), img);

            int imgLabel = strLabels.indexOf(data[1]);

            labelsBuf.put(counter.get(), imgLabel);

            counter.getAndIncrement();
        });

        FaceRecognizer faceRecognizer = LBPHFaceRecognizer.create();

        faceRecognizer.train(images, labels);

        strLabels.forEach(label -> {
            faceRecognizer.setLabelInfo(strLabels.indexOf(label), new String(label.getBytes(UTF_8)));
        });

        java.nio.file.Path tmp = java.nio.file.Files.createTempFile("trained_dataset", ".xml");
        faceRecognizer.write(tmp.toString());
        byte[] xmlBytes = java.nio.file.Files.readAllBytes(tmp);
        java.nio.file.Files.deleteIfExists(tmp);

        // Save to database with retry logic for SQLite lock contention
        saveTrainedDatasetWithRetry(xmlBytes);

        matUtil.clearMatVector(images);

        Map<String, String> personHashes = new java.util.HashMap<>();

        // Properly close the stream to avoid resource leaks
        try (var stream = Files.list(rootFolderPath)) {
            stream.filter(path -> path.toFile().isDirectory())
                .forEach(personFolder -> {
                    String personName = personFolder.getFileName().toString();
                    try {
                        String hash = computeFolderHash(personFolder);
                        personHashes.put(personName, hash);
                    } catch (IOException e) {
                        log.warn("Failed to compute hash for folder {}", personFolder, e);
                    }
                });
        }

        // Save metadata with retry logic
        saveTrainingMetadataWithRetry(personHashes);

        return faceRecognizer;
    }

    /**
     * Save trained dataset to database with retry logic for lock contention
     */
    private void saveTrainedDatasetWithRetry(byte[] xmlBytes) {
        int maxRetries = 5;
        int retryDelayMs = 200;

        for (int attempt = 1; attempt <= maxRetries; attempt++) {
            try {
                trainedDatasetRepository.deleteAll();
                TrainedDataset trained = new TrainedDataset();
                trained.setModelXml(xmlBytes);
                trainedDatasetRepository.save(trained);
                log.debug("Successfully saved trained dataset to database on attempt {}", attempt);
                return; // Success
            } catch (org.springframework.dao.CannotAcquireLockException |
                     org.springframework.dao.TransientDataAccessResourceException e) {
                if (attempt < maxRetries) {
                    long delay = retryDelayMs * (long) Math.pow(2, attempt - 1);
                    log.warn("Database locked while saving trained dataset (attempt {}/{}). Retrying in {}ms...",
                            attempt, maxRetries, delay);
                    try {
                        Thread.sleep(delay);
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                        throw new RuntimeException("Interrupted while retrying database save", ie);
                    }
                } else {
                    log.error("Failed to save trained dataset after {} attempts due to database lock", maxRetries);
                    throw new RuntimeException("Database is locked - failed to save trained dataset after retries", e);
                }
            }
        }
    }

    /**
     * Save training metadata to database with retry logic for lock contention
     */
    private void saveTrainingMetadataWithRetry(Map<String, String> personHashes) {
        int maxRetries = 5;
        int retryDelayMs = 200;

        for (int attempt = 1; attempt <= maxRetries; attempt++) {
            try {
                trainingMetadataRepository.deleteAll();
                personHashes.forEach((personName, hash) -> {
                    TrainingMetadata metadata = new TrainingMetadata();
                    metadata.setPersonName(personName);
                    metadata.setFolderHash(hash);
                    trainingMetadataRepository.save(metadata);
                });
                log.debug("Successfully saved training metadata to database on attempt {}", attempt);
                return; // Success
            } catch (org.springframework.dao.CannotAcquireLockException |
                     org.springframework.dao.TransientDataAccessResourceException e) {
                if (attempt < maxRetries) {
                    long delay = retryDelayMs * (long) Math.pow(2, attempt - 1);
                    log.warn("Database locked while saving training metadata (attempt {}/{}). Retrying in {}ms...",
                            attempt, maxRetries, delay);
                    try {
                        Thread.sleep(delay);
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                        throw new RuntimeException("Interrupted while retrying database save", ie);
                    }
                } else {
                    log.error("Failed to save training metadata after {} attempts due to database lock", maxRetries);
                    throw new RuntimeException("Database is locked - failed to save training metadata after retries", e);
                }
            }
        }
    }

    public boolean isTrainingDataChanged(Path rootFolder) throws IOException {
        Path rootFolderPath = rootFolder;

        log.info("Checking for training data changes in: {}", rootFolderPath);

        Map<String, String> currentHashes = new HashMap<>();

        // Properly close the stream to avoid resource leaks
        try (var stream = Files.list(rootFolderPath)) {
            stream.filter(path -> path.toFile().isDirectory())
                .forEach(personFolder -> {
                    String personName = personFolder.getFileName().toString();
                    try {
                        String hash = computeFolderHash(personFolder);
                        currentHashes.put(personName, hash);
                        log.debug("Computed hash for {}: {}", personName, hash.substring(0, 8) + "...");
                    } catch (IOException e) {
                        log.warn("Failed to compute hash for folder {}", personFolder, e);
                    }
                });
        }

        log.info("Found {} person folders in training directory", currentHashes.size());

        List<TrainingMetadata> allMetadata = trainingMetadataRepository.findAll();
        Map<String, String> storedHashes = allMetadata.stream()
                .collect(Collectors.toMap(TrainingMetadata::getPersonName, TrainingMetadata::getFolderHash, (a, b) -> b));

        log.info("Found {} stored person metadata records in database", storedHashes.size());

        if (currentHashes.size() != storedHashes.size()) {
            log.warn("Training data changed: Number of person folders changed from {} to {}",
                storedHashes.size(), currentHashes.size());

            Set<String> addedPeople = new HashSet<>(currentHashes.keySet());
            addedPeople.removeAll(storedHashes.keySet());
            if (!addedPeople.isEmpty()) {
                log.info("New person folders detected: {}", addedPeople);
            }

            Set<String> removedPeople = new HashSet<>(storedHashes.keySet());
            removedPeople.removeAll(currentHashes.keySet());
            if (!removedPeople.isEmpty()) {
                log.info("Person folders removed: {}", removedPeople);
            }

            return true;
        }

        for (Map.Entry<String, String> entry : currentHashes.entrySet()) {
            String personName = entry.getKey();
            String currentHash = entry.getValue();
            String storedHash = storedHashes.get(personName);
            if (storedHash == null || !storedHash.equals(currentHash)) {
                log.warn("Training data changed: Images changed for person '{}'", personName);
                log.debug("  Current hash: {}", currentHash.substring(0, 16) + "...");
                log.debug("  Stored hash:  {}", storedHash != null ? storedHash.substring(0, 16) + "..." : "null");
                return true;
            }
        }

        log.info("No changes detected in training data");
        return false;
    }

    public FaceRecognizer ensureTrained(Path rootFolder) throws IOException {
        Path rootFolderPath = rootFolder;

        log.info("=== Starting face recognition model check ===");
        log.info("Training folder: {}", rootFolderPath.toAbsolutePath());

        boolean datasetExists = !trainedDatasetRepository.findAll().isEmpty();
        log.info("Trained model exists in database: {}", datasetExists);

        if (!datasetExists) {
            log.warn("No trained model found in database - initial training required");
        }

        boolean changed = !datasetExists || isTrainingDataChanged(rootFolderPath);

        if (changed) {
            log.warn("=== RETRAINING TRIGGERED ===");
            log.info("Reason: {}", !datasetExists ? "No existing model" : "Training data changed");
            log.info("Starting model training process...");
            FaceRecognizer recognizer = train(rootFolderPath);
            log.info("=== TRAINING COMPLETED SUCCESSFULLY ===");
            return recognizer;
        } else {
            log.info("Training data unchanged; loading existing model from DB");
            FaceRecognizer recognizer = load();
            log.info("=== MODEL LOADED FROM DATABASE ===");
            return recognizer;
        }
    }

    private String computeFolderHash(Path folder) throws IOException {
        log.debug("Computing hash for folder: {}", folder.getFileName());

        java.security.MessageDigest digest;
        try {
            digest = java.security.MessageDigest.getInstance("SHA-256");
        } catch (java.security.NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 not available", e);
        }

        long[] fileCount = {0};

        // Properly close the stream to avoid resource leaks
        try (var stream = Files.walk(folder)) {
            stream.filter(Files::isRegularFile)
                .sorted()
                .forEach(path -> {
                    try {
                        fileCount[0]++;
                        digest.update(path.toString().getBytes(java.nio.charset.StandardCharsets.UTF_8));
                        digest.update(java.nio.file.Files.readAllBytes(path));
                        log.trace("  Processed file: {}", path.getFileName());
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                });
        }

        log.debug("  Total files processed for {}: {}", folder.getFileName(), fileCount[0]);

        byte[] hashBytes = digest.digest();
        StringBuilder sb = new StringBuilder();
        for (byte b : hashBytes) {
            sb.append(String.format("%02x", b));
        }
        return sb.toString();
    }

    /**
     * Calculate adaptive distance threshold based on face size ratio to frame area.
     * This makes the recognition resolution-independent and adjusts for distance to camera.
     *
     * Logic:
     * - Larger faces (closer to camera) = stricter threshold (better quality expected)
     * - Smaller faces (farther from camera) = more lenient threshold (account for lower quality)
     *
     * @param faceRatio Ratio of face area to frame area (0.0 to 1.0)
     * @return Adaptive distance threshold for recognition
     */
    private double calculateAdaptiveThreshold(double faceRatio) {
        if (faceRatio >= EXPECTED_FACE_RATIO) {
            // Face is larger than expected (person is closer to camera)
            // Use stricter threshold - high quality image should match better
            // Example: face is 10% of frame (2x expected) → threshold = 30
            //          face is 7.5% of frame (1.5x expected) → threshold = 40
            double threshold = BASE_DISTANCE * (EXPECTED_FACE_RATIO / faceRatio);
            return Math.max(threshold, MIN_ADAPTIVE_THRESHOLD);
        } else {
            // Face is smaller than expected (person is farther from camera)
            // Use more lenient threshold - lower quality may affect matching
            // Example: face is 2.5% of frame (0.5x expected) → threshold = 75
            //          face is 1.25% of frame (0.25x expected) → threshold = 85
            double factor = Math.sqrt(EXPECTED_FACE_RATIO / faceRatio);
            double threshold = BASE_DISTANCE * factor;
            return Math.min(threshold, MAX_ADAPTIVE_THRESHOLD);
        }
    }

    private String sanitizeLabel(String label) {
        if (label == null) {
            return UNKNOWN;
        }

        String sanitized = label.replace("\u0000", "").trim();
        return sanitized.isEmpty() ? UNKNOWN : sanitized;
    }
}
