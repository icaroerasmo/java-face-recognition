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
import java.util.HashMap;
import java.util.List;
import java.util.Map;
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

    public static final int MIN_SCORE = 40;
    public static final String UNKNOWN = "Unknown";

    public FaceRecognizer load() {
        FaceRecognizer faceRecognizer = LBPHFaceRecognizer.create();
        // Load the latest trained dataset XML from DB
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

    public FaceRecognition test(FaceRecognizer faceRecognizer, String testFile) throws Exception {
        final Mat testImage = imread(testFile/*,IMREAD_GRAYSCALE*/);
        return test(faceRecognizer, testImage);
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
                String detectedPerson = label.substring(0, label.length() - 1); // Remove the last character which is a space
                double detectionConfidence = confidencePtr.get(0);

                if (detectionConfidence > MIN_SCORE) {
                    log.debug("Detected person is {} with confidence {}" +
                                    " but score is bigger than {} so result is {}.",
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

                    Mat img = imread(image.getAbsolutePath() /*, IMREAD_GRAYSCALE*/);

                    List<Rect> facesList = deepLearningFaceDetectionService.detect(img);

                    if (facesList.isEmpty()) {
                        return null;
                    }

                    Rect faceRect = facesList.get(0);
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

//        FaceRecognizer faceRecognizer = FisherFaceRecognizer.create();
//         FaceRecognizer faceRecognizer = EigenFaceRecognizer.create();
        FaceRecognizer faceRecognizer = LBPHFaceRecognizer.create();

        faceRecognizer.train(images, labels);

        strLabels.forEach(label -> {
            faceRecognizer.setLabelInfo(strLabels.indexOf(label), new String(label.getBytes(UTF_8)));
        });

        // Persist the trained model XML into database instead of filesystem
        java.nio.file.Path tmp = java.nio.file.Files.createTempFile("trained_dataset", ".xml");
        faceRecognizer.write(tmp.toString());
        byte[] xmlBytes = java.nio.file.Files.readAllBytes(tmp);
        java.nio.file.Files.deleteIfExists(tmp);

        trainedDatasetRepository.deleteAll();
        TrainedDataset trained = new TrainedDataset();
        trained.setModelXml(xmlBytes);
        trainedDatasetRepository.save(trained);

        matUtil.clearMatVector(images);

        // Compute and store training metadata per person folder
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

        personHashes.forEach((personName, hash) -> {
            TrainingMetadata metadata = new TrainingMetadata();
            metadata.setPersonName(personName);
            metadata.setFolderHash(hash);
            trainingMetadataRepository.save(metadata);
        });

        return faceRecognizer;
    }

    /**
     * Checks whether the training data (person folders and images) has changed
     * compared to what is stored in the training_metadata table.
     */
    public boolean isTrainingDataChanged(Path rootFolder) throws IOException {
        Path rootFolderPath = rootFolder;

        // Compute current hashes per person folder
        Map<String, String> currentHashes = new HashMap<>();
        Files.list(rootFolderPath)
                .filter(path -> path.toFile().isDirectory())
                .forEach(personFolder -> {
                    String personName = personFolder.getFileName().toString();
                    try {
                        String hash = computeFolderHash(personFolder);
                        currentHashes.put(personName, hash);
                    } catch (IOException e) {
                        log.warn("Failed to compute hash for folder {}", personFolder, e);
                    }
                });

        // Load stored hashes from metadata
        List<TrainingMetadata> allMetadata = trainingMetadataRepository.findAll();
        Map<String, String> storedHashes = allMetadata.stream()
                .collect(Collectors.toMap(TrainingMetadata::getPersonName, TrainingMetadata::getFolderHash, (a, b) -> b));

        // If the number of persons differs, training data changed
        if (currentHashes.size() != storedHashes.size()) {
            return true;
        }

        // Compare hash per person
        for (Map.Entry<String, String> entry : currentHashes.entrySet()) {
            String personName = entry.getKey();
            String currentHash = entry.getValue();
            String storedHash = storedHashes.get(personName);
            if (storedHash == null || !storedHash.equals(currentHash)) {
                return true;
            }
        }

        return false;
    }

    /**
     * Convenience method that checks for training data changes and retrains
     * the model (and metadata + DB-stored XML) if needed.
     */
    public FaceRecognizer ensureTrained(Path rootFolder) throws IOException {
        Path rootFolderPath = rootFolder;

        // datasetExists: do we have a trained model stored in the DB?
        boolean datasetExists = !trainedDatasetRepository.findAll().isEmpty();
        boolean changed = !datasetExists || isTrainingDataChanged(rootFolderPath);

        if (changed) {
            log.info("Training data changed or trained dataset missing; retraining and updating DB model");
            // train(...) will retrain the recognizer, overwrite the trained_dataset row,
            // and refresh folder hashes in training_metadata
            return train(rootFolderPath);
        } else {
            log.info("Training data unchanged; loading existing model from DB");
            return load();
        }
    }

    private String computeFolderHash(Path folder) throws IOException {
        java.security.MessageDigest digest;
        try {
            digest = java.security.MessageDigest.getInstance("SHA-256");
        } catch (java.security.NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 not available", e);
        }

        Files.walk(folder)
                .filter(Files::isRegularFile)
                .sorted()
                .forEach(path -> {
                    try {
                        digest.update(path.toString().getBytes(java.nio.charset.StandardCharsets.UTF_8));
                        digest.update(java.nio.file.Files.readAllBytes(path));
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                });

        byte[] hashBytes = digest.digest();
        StringBuilder sb = new StringBuilder();
        for (byte b : hashBytes) {
            sb.append(String.format("%02x", b));
        }
        return sb.toString();
    }
}
