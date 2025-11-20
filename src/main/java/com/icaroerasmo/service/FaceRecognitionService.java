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

    public static final int MIN_SCORE = 50;
    public static final String UNKNOWN = "Unknown";

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

    public FaceRecognition test(FaceRecognizer faceRecognizer, Mat testImage) {

        List<FaceRecognition.DetectedFaces> detectedFaces = deepLearningFaceDetectionService.detect(testImage).stream().map(faceRect -> {

            Mat img = null;

            try {
                img = matUtil.convertToGray(new Mat(testImage, faceRect));

                IntPointer detectedPersonPtr = new IntPointer(1);
                DoublePointer confidencePtr = new DoublePointer(1);

                faceRecognizer.predict(img, detectedPersonPtr, confidencePtr);

                String label = faceRecognizer.getLabelInfo(detectedPersonPtr.get(0)).getString();
                String detectedPerson = label.substring(0, label.length() - 1);
                double detectionConfidence = confidencePtr.get(0);

                if (detectionConfidence > MIN_SCORE) {
                    log.debug("Detected person is {} with confidence {} but score is bigger than {} so result is {}.",
                            detectedPerson, detectionConfidence, MIN_SCORE, UNKNOWN);
                    detectedPerson = UNKNOWN;
                } else {
                    log.debug("Detected person is {} with confidence {}", detectedPerson, detectionConfidence);
                }

                return new FaceRecognition.DetectedFaces(detectedPerson, detectionConfidence, faceRect);
            } catch(Exception e) {
                log.error("Error processing face detection", e);
                throw new RuntimeException("Error processing face detection", e);
            } finally {
                matUtil.releaseResources(img);
            }
        }).filter(detected -> detected != null).toList();

        return new FaceRecognition(detectedFaces, testImage);
    }

    public FaceRecognizer train(Path rootFolder) throws IOException {
        Path rootFolderPath = rootFolder;
        Map<Path, Object[]> fileList = Files.list(rootFolderPath)
                .filter(file -> file.toFile().isDirectory())
                .flatMap(folder -> {
                    try {
                        var personName = folder.getName(folder.getNameCount() - 1).toString();
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

        trainedDatasetRepository.deleteAll();
        TrainedDataset trained = new TrainedDataset();
        trained.setModelXml(xmlBytes);
        trainedDatasetRepository.save(trained);

        matUtil.clearMatVector(images);

        Map<String, String> personHashes = new java.util.HashMap<>();
        Files.list(rootFolderPath)
                .filter(path -> path.toFile().isDirectory())
                .forEach(personFolder -> {
                    String personName = personFolder.getFileName().toString();
                    try {
                        String hash = computeFolderHash(personFolder);
                        personHashes.put(personName, hash);
                    } catch (IOException e) {
                        log.warn("Failed to compute hash for folder {}", personFolder, e);
                    }
                });

        trainingMetadataRepository.deleteAll();
        personHashes.forEach((personName, hash) -> {
            TrainingMetadata metadata = new TrainingMetadata();
            metadata.setPersonName(personName);
            metadata.setFolderHash(hash);
            trainingMetadataRepository.save(metadata);
        });

        return faceRecognizer;
    }

    public boolean isTrainingDataChanged(Path rootFolder) throws IOException {
        Path rootFolderPath = rootFolder;

        log.info("Checking for training data changes in: {}", rootFolderPath);

        Map<String, String> currentHashes = new HashMap<>();
        Files.list(rootFolderPath)
                .filter(path -> path.toFile().isDirectory())
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
        Files.walk(folder)
                .filter(Files::isRegularFile)
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

        log.debug("  Total files processed for {}: {}", folder.getFileName(), fileCount[0]);

        byte[] hashBytes = digest.digest();
        StringBuilder sb = new StringBuilder();
        for (byte b : hashBytes) {
            sb.append(String.format("%02x", b));
        }
        return sb.toString();
    }
}
