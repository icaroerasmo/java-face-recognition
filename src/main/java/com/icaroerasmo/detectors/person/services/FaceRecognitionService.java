package com.icaroerasmo.detectors.person.services;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.icaroerasmo.model.FaceRecognition;
import com.icaroerasmo.repository.TrainedDatasetRepository;
import com.icaroerasmo.repository.TrainingMetadataRepository;
import com.icaroerasmo.repository.entity.TrainedDataset;
import com.icaroerasmo.repository.entity.TrainingMetadata;
import com.icaroerasmo.utils.MatUtil;
import com.icaroerasmo.utils.OpenCvResourceHelper;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_objdetect.FaceRecognizerSF;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;

import static com.icaroerasmo.utils.Constants.DESIRED_SCORE;
import static org.bytedeco.opencv.global.opencv_core.CV_32FC1;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;

@Log4j2
@Service
@RequiredArgsConstructor
public class FaceRecognitionService {

    private static final String SFACE_MODEL_FILE = "opencv/face_recognition_sface_2021dec.onnx";
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();
    private static final TypeReference<StoredGallery> GALLERY_TYPE = new TypeReference<>() {};

    // Normalized distance (1 - cosine similarity). Lower = better.
    public static final double BASE_DISTANCE = 0.55;
    public static final String UNKNOWN = "Unknown";

    // Expected face ratio for calibration (5% of frame area is typical for good recognition)
    private static final double EXPECTED_FACE_RATIO = 0.05;

    // Min and max adaptive thresholds for normalized distance.
    private static final double MIN_ADAPTIVE_THRESHOLD = 0.45;
    private static final double MAX_ADAPTIVE_THRESHOLD = 0.60;

    private final DeepLearningFaceDetectionService deepLearningFaceDetectionService;
    private final TrainingMetadataRepository trainingMetadataRepository;
    private final TrainedDatasetRepository trainedDatasetRepository;

    public FaceRecognitionRuntime load() {
        TrainedDataset trained = trainedDatasetRepository.findAll().stream().findFirst()
            .orElseThrow(() -> new IllegalStateException("No trained dataset found in database"));

        StoredGallery storedGallery = deserializeGallery(trained.getModelXml());
        FaceRecognizerSF recognizer = createRecognizer();
        Map<String, List<Mat>> galleryFeatures = buildRuntimeGallery(storedGallery.embeddings());
        return new FaceRecognitionRuntime(recognizer, galleryFeatures);
    }

    /**
     * Test face recognition on an image.
     */
    public FaceRecognition test(FaceRecognitionRuntime faceRecognitionRuntime, Mat testImage) {

        if (testImage == null || testImage.empty()) {
            log.warn("Cannot perform face recognition on null or empty image");
            return new FaceRecognition(List.of());
        }

        final double frameArea = testImage.rows() * testImage.cols();
        log.debug("Frame resolution: {}x{}, total area: {} pixels",
            testImage.cols(), testImage.rows(), (long) frameArea);

        List<FaceRecognition.DetectedFaces> detectedFaces = deepLearningFaceDetectionService.detect(testImage).stream()
            .map(faceDetection -> detectFace(faceRecognitionRuntime, testImage, faceDetection, frameArea))
            .filter(Objects::nonNull)
            .toList();

        return new FaceRecognition(detectedFaces);
    }

    private FaceRecognition.DetectedFaces detectFace(
        FaceRecognitionRuntime faceRecognitionRuntime,
        Mat testImage,
        DeepLearningFaceDetectionService.FaceDetection faceDetection,
        double frameArea
    ) {
        Mat faceBox = null;
        Mat alignedFace = null;
        Mat feature = null;

        try {
            faceBox = createFaceBoxMat(faceDetection);
            alignedFace = new Mat();
            feature = new Mat();

            synchronized (faceRecognitionRuntime.getRecognizer()) {
                faceRecognitionRuntime.getRecognizer().alignCrop(testImage, faceBox, alignedFace);
                faceRecognitionRuntime.getRecognizer().feature(alignedFace, feature);
            }

            if (alignedFace.empty() || feature.empty()) {
                log.warn("Failed to align or extract features for face at x={}, y={}, width={}, height={}",
                    faceDetection.rect().x(), faceDetection.rect().y(), faceDetection.rect().width(), faceDetection.rect().height());
                return null;
            }

            MatchResult matchResult = matchAgainstGallery(faceRecognitionRuntime, feature);
            double detectionDistance = matchResult.distance();
            String detectedPerson = matchResult.personName();

            double faceArea = faceDetection.rect().width() * faceDetection.rect().height();
            double faceRatio = faceArea / frameArea;
            double adaptiveThreshold = calculateAdaptiveThreshold(faceRatio);

            log.debug("Detected person is {} with normalized distance {} (adaptive threshold: {}, face ratio: {}%)",
                    detectedPerson,
                    String.format("%.4f", detectionDistance),
                    String.format("%.4f", adaptiveThreshold),
                    String.format("%.4f", faceRatio * 100));

            return new FaceRecognition.DetectedFaces(detectedPerson, detectionDistance, MatUtil.cloneRect(faceDetection.rect()));
        } catch (Exception e) {
            log.error("Error processing face detection for rect: x={}, y={}, width={}, height={}",
                faceDetection.rect().x(), faceDetection.rect().y(), faceDetection.rect().width(), faceDetection.rect().height(), e);
            return null;
        } finally {
            MatUtil.releaseResources(faceBox, alignedFace, feature);
            if (faceDetection.rect() != null) {
                faceDetection.rect().deallocate();
            }
        }
    }

    @Transactional
    public FaceRecognitionRuntime train(Path rootFolder) throws IOException {
        Path rootFolderPath = rootFolder;
        List<Path> personFolders;

        try (var rootStream = Files.list(rootFolderPath)) {
            personFolders = rootStream
                .filter(Files::isDirectory)
                .sorted()
                .toList();
        }

        FaceRecognizerSF recognizer = createRecognizer();
        Map<String, List<Mat>> runtimeGallery = new HashMap<>();
        Map<String, List<float[]>> storedGallery = new HashMap<>();
        List<String> invalidPeople = new ArrayList<>();

        try {
            for (Path personFolder : personFolders) {
                String personName = personFolder.getFileName().toString();
                List<Mat> personRuntimeEmbeddings = new ArrayList<>();
                List<float[]> personStoredEmbeddings = new ArrayList<>();

                try (var personImagesStream = Files.list(personFolder)) {
                    List<Path> personImages = personImagesStream
                        .filter(Files::isRegularFile)
                        .sorted()
                        .toList();

                    for (Path imagePath : personImages) {
                        Mat img = imread(imagePath.toString());
                        Mat faceBox = null;
                        Mat alignedFace = null;
                        Mat feature = null;

                        try {
                            if (img.empty()) {
                                log.warn("Skipping unreadable training image '{}' for '{}'", imagePath.getFileName(), personName);
                                continue;
                            }

                            List<DeepLearningFaceDetectionService.FaceDetection> facesList = deepLearningFaceDetectionService.detect(img);

                            if (facesList.isEmpty()) {
                                log.warn("Skipping training image '{}' for '{}': no face detected", imagePath.getFileName(), personName);
                                continue;
                            }

                            if (facesList.size() > 1) {
                                log.warn("Skipping training image '{}' for '{}': {} faces detected; training requires exactly one face per image",
                                    imagePath.getFileName(), personName, facesList.size());
                                continue;
                            }

                            faceBox = createFaceBoxMat(facesList.getFirst());
                            alignedFace = new Mat();
                            feature = new Mat();

                            synchronized (recognizer) {
                                recognizer.alignCrop(img, faceBox, alignedFace);
                                recognizer.feature(alignedFace, feature);
                            }

                            if (alignedFace.empty() || feature.empty()) {
                                log.warn("Skipping training image '{}' for '{}': failed to extract face embedding",
                                    imagePath.getFileName(), personName);
                                continue;
                            }

                            personRuntimeEmbeddings.add(feature.clone());
                            personStoredEmbeddings.add(extractFeatureVector(feature));
                        } finally {
                            MatUtil.releaseResources(img, faceBox, alignedFace, feature);
                        }
                    }
                }

                if (personStoredEmbeddings.isEmpty()) {
                    invalidPeople.add(personName);
                    releaseGalleryFeatures(personRuntimeEmbeddings);
                    continue;
                }

                runtimeGallery.put(personName, personRuntimeEmbeddings);
                storedGallery.put(personName, personStoredEmbeddings);
            }

            if (!invalidPeople.isEmpty()) {
                throw new IllegalStateException(
                    "Training requires at least one single-face image per person. No valid training samples found for: "
                        + String.join(", ", invalidPeople)
                );
            }

            if (storedGallery.isEmpty()) {
                throw new IllegalStateException("No valid training images found. Each image must contain exactly one detectable face.");
            }

            saveTrainedDatasetWithRetry(serializeGallery(storedGallery));
            saveTrainingMetadataWithRetry(computePersonHashes(rootFolderPath));

            return new FaceRecognitionRuntime(recognizer, runtimeGallery);
        } catch (IOException | RuntimeException e) {
            releaseRuntimeGallery(runtimeGallery);
            try {
                recognizer.close();
            } catch (Exception ignore) {
            }
            throw e;
        }
    }

    /**
     * Save trained dataset to database with retry logic for lock contention
     */
    private void saveTrainedDatasetWithRetry(byte[] datasetBytes) {
        int maxRetries = 5;
        int retryDelayMs = 200;

        for (int attempt = 1; attempt <= maxRetries; attempt++) {
            try {
                trainedDatasetRepository.deleteAll();
                TrainedDataset trained = new TrainedDataset();
                trained.setModelXml(datasetBytes);
                trainedDatasetRepository.save(trained);
                log.debug("Successfully saved trained dataset to database on attempt {}", attempt);
                return;
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
                return;
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

        Map<String, String> currentHashes = computePersonHashes(rootFolderPath);

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

    public FaceRecognitionRuntime ensureTrained(Path rootFolder) throws IOException {
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
            FaceRecognitionRuntime recognizer = train(rootFolderPath);
            log.info("=== TRAINING COMPLETED SUCCESSFULLY ===");
            return recognizer;
        }

        try {
            log.info("Training data unchanged; loading existing model from DB");
            FaceRecognitionRuntime recognizer = load();
            log.info("=== MODEL LOADED FROM DATABASE ===");
            return recognizer;
        } catch (IllegalStateException e) {
            log.warn("Stored face-recognition dataset is incompatible or unreadable. Retraining from filesystem.", e);
            FaceRecognitionRuntime recognizer = train(rootFolderPath);
            log.info("=== TRAINING COMPLETED SUCCESSFULLY ===");
            return recognizer;
        }
    }

    private Map<String, String> computePersonHashes(Path rootFolderPath) throws IOException {
        Map<String, String> personHashes = new HashMap<>();

        try (var stream = Files.list(rootFolderPath)) {
            stream.filter(Files::isDirectory)
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

        return personHashes;
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
     * Calculate adaptive threshold based on face size ratio to frame area.
     */
    private double calculateAdaptiveThreshold(double faceRatio) {
        if (faceRatio >= EXPECTED_FACE_RATIO) {
            double threshold = BASE_DISTANCE * (EXPECTED_FACE_RATIO / faceRatio);
            return Math.max(threshold, MIN_ADAPTIVE_THRESHOLD);
        }

        double factor = Math.sqrt(EXPECTED_FACE_RATIO / faceRatio);
        double threshold = BASE_DISTANCE * factor;
        return Math.min(threshold, MAX_ADAPTIVE_THRESHOLD);
    }

    private MatchResult matchAgainstGallery(FaceRecognitionRuntime runtime, Mat probeFeature) {
        String bestPerson = UNKNOWN;
        double bestSimilarity = Double.NEGATIVE_INFINITY;

        synchronized (runtime.getRecognizer()) {
            for (Map.Entry<String, List<Mat>> entry : runtime.getGalleryFeatures().entrySet()) {
                for (Mat candidateFeature : entry.getValue()) {
                    double similarity = runtime.getRecognizer().match(probeFeature, candidateFeature, FaceRecognizerSF.FR_COSINE);
                    if (Double.isNaN(similarity)) {
                        continue;
                    }

                    if (similarity > bestSimilarity) {
                        bestSimilarity = similarity;
                        bestPerson = entry.getKey();
                    }
                }
            }
        }

        if (bestSimilarity == Double.NEGATIVE_INFINITY) {
            return new MatchResult(UNKNOWN, 1.0);
        }

        return new MatchResult(bestPerson, 1.0 - clampSimilarity(bestSimilarity));
    }

    private Mat createFaceBoxMat(DeepLearningFaceDetectionService.FaceDetection detection) {
        Mat faceBox = new Mat(1, 15, CV_32FC1);
        FloatIndexer indexer = faceBox.createIndexer();

        try {
            indexer.put(0, 0, detection.rect().x());
            indexer.put(0, 1, detection.rect().y());
            indexer.put(0, 2, detection.rect().width());
            indexer.put(0, 3, detection.rect().height());

            for (int landmarkIndex = 0; landmarkIndex < detection.landmarks().length; landmarkIndex++) {
                indexer.put(0, 4 + landmarkIndex, detection.landmarks()[landmarkIndex]);
            }

            indexer.put(0, 14, detection.confidence());
            return faceBox;
        } finally {
            indexer.release();
        }
    }

    private float[] extractFeatureVector(Mat feature) {
        FloatIndexer indexer = feature.createIndexer();
        try {
            int featureLength = (int) feature.total();
            float[] values = new float[featureLength];
            for (int index = 0; index < featureLength; index++) {
                values[index] = feature.dims() > 1 ? indexer.get(0, index) : indexer.get(index);
            }
            return values;
        } finally {
            indexer.release();
        }
    }

    private Map<String, List<Mat>> buildRuntimeGallery(Map<String, List<float[]>> storedEmbeddings) {
        if (storedEmbeddings == null || storedEmbeddings.isEmpty()) {
            throw new IllegalStateException("Stored face gallery is empty");
        }

        Map<String, List<Mat>> gallery = new HashMap<>();
        storedEmbeddings.forEach((personName, embeddings) -> {
            List<Mat> personFeatures = new ArrayList<>();
            for (float[] embedding : embeddings) {
                personFeatures.add(createFeatureMat(embedding));
            }
            gallery.put(personName, personFeatures);
        });
        return gallery;
    }

    private Mat createFeatureMat(float[] embedding) {
        if (embedding == null || embedding.length == 0) {
            throw new IllegalStateException("Encountered empty face embedding in stored gallery");
        }

        Mat feature = new Mat(1, embedding.length, CV_32FC1);
        FloatIndexer indexer = feature.createIndexer();
        try {
            for (int index = 0; index < embedding.length; index++) {
                indexer.put(0, index, embedding[index]);
            }
            return feature;
        } finally {
            indexer.release();
        }
    }

    private byte[] serializeGallery(Map<String, List<float[]>> gallery) {
        try {
            return OBJECT_MAPPER.writeValueAsBytes(new StoredGallery(gallery));
        } catch (IOException e) {
            throw new IllegalStateException("Failed to serialize face gallery", e);
        }
    }

    private StoredGallery deserializeGallery(byte[] rawBytes) {
        try {
            StoredGallery gallery = OBJECT_MAPPER.readValue(rawBytes, GALLERY_TYPE);
            if (gallery == null || gallery.embeddings() == null || gallery.embeddings().isEmpty()) {
                throw new IllegalStateException("Stored face gallery is empty");
            }
            return gallery;
        } catch (IOException e) {
            throw new IllegalStateException("Failed to deserialize stored face gallery", e);
        }
    }

    private FaceRecognizerSF createRecognizer() {
        try {
            String modelPath = OpenCvResourceHelper.getResourcePath(SFACE_MODEL_FILE, FaceRecognitionService.class);
            return FaceRecognizerSF.create(modelPath, "");
        } catch (Exception e) {
            throw new IllegalStateException("Failed to initialize FaceRecognizerSF", e);
        }
    }

    private void releaseRuntimeGallery(Map<String, List<Mat>> runtimeGallery) {
        runtimeGallery.values().forEach(this::releaseGalleryFeatures);
    }

    private void releaseGalleryFeatures(List<Mat> features) {
        features.forEach(feature -> {
            if (feature != null) {
                feature.release();
            }
        });
    }

    private double clampSimilarity(double similarity) {
        return Math.max(0.0, Math.min(1.0, similarity));
    }

    private record MatchResult(String personName, double distance) {
    }

    private record StoredGallery(Map<String, List<float[]>> embeddings) {
    }
}
