package com.icaroerasmo.service;

import com.icaroerasmo.model.FaceRecognition;
import com.icaroerasmo.utils.MatUtil;
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

    private static final Path DATASET = Paths.get("trained_dataset.xml");
    public static final int MIN_SCORE = 40;
    public static final String UNKNOWN = "Unknown";

    private final DeepLearningFaceDetectionService deepLearningFaceDetectionService;
    private final MatUtil matUtil;

    public FaceRecognizer load() {
        FaceRecognizer faceRecognizer = LBPHFaceRecognizer.create();
        faceRecognizer.read(DATASET.toString());
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
                    log.info("Detected person is {} with confidence {}", detectedPerson, detectionConfidence);
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

    public FaceRecognizer train(String root) throws IOException {
        Path rootFolder = Paths.get(root);
        Map<Path, Object[]> fileList = Files.list(rootFolder).
                filter(file -> file.toFile().isDirectory()).
                flatMap(folder -> {
                    try {
                        var personName = folder.getName(folder.getNameCount()-1).toString();
                        return Files.list(folder).map(file -> Map.entry(file, personName));
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                }).
                map(entry -> {
                    File image = entry.getKey().toFile();

                    Mat img = imread(image.getAbsolutePath()/*, IMREAD_GRAYSCALE*/);

                    List<Rect> facesList = deepLearningFaceDetectionService.detect(img);

                    if(facesList.isEmpty()) {
                        return null;
                    }

                    Rect faceRect = facesList.get(0);
                    Mat face = new Mat(img, faceRect);

                    matUtil.releaseResources(img);

                    return Map.entry(entry.getKey(), new Object[]{face, entry.getValue()});
                }).
                filter(entry -> entry != null).
                collect(toMap(Map.Entry::getKey, Map.Entry::getValue));

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

        faceRecognizer.write(DATASET.toString());

        matUtil.clearMatVector(images);

        return faceRecognizer;
    }
}
