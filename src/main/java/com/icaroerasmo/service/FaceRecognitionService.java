package com.icaroerasmo.service;

import com.icaroerasmo.model.FaceRecognition;
import com.icaroerasmo.utils.MatUtil;
import com.icaroerasmo.properties.TrainingProperties;
import com.icaroerasmo.repository.TrainingMetadataRepository;
import com.icaroerasmo.repository.entity.TrainingMetadata;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.bytedeco.opencv.opencv_core.*;
import org.jetbrains.annotations.NotNull;
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
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

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

    public static final int MIN_SCORE = 40;
    public static final String UNKNOWN = "Unknown";

    private Path getDatasetPath() {
        return Paths.get(trainingProperties.getDatasetPath());
    }

    public FaceRecognizer load() {
        FaceRecognizer faceRecognizer = LBPHFaceRecognizer.create();
        faceRecognizer.read(getDatasetPath().toString());
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

        faceRecognizer.write(getDatasetPath().toString());
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
