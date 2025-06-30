package com.icaroerasmo.service;

import com.icaroerasmo.model.FaceRecognition;
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
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.stream.Collectors.toMap;
import static org.bytedeco.opencv.global.opencv_core.CV_32SC1;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

@Log4j2
@Service
@RequiredArgsConstructor
public class FaceRecognitionService {

    private static final Path DATASET = Paths.get("trained_dataset.xml");

    private final DeepLearningFaceDetectionService deepLearningFaceDetectionService;

    public FaceRecognizer load() {
        FaceRecognizer faceRecognizer = LBPHFaceRecognizer.create();
        faceRecognizer.read(DATASET.toString());
        return faceRecognizer;
    }

    public FaceRecognition test(FaceRecognizer faceRecognizer, String testFile) throws Exception {

        final Mat testImage = imread(testFile/*,IMREAD_GRAYSCALE*/);

        List<FaceRecognition.DetectedFaces> detectedFaces = deepLearningFaceDetectionService.detect(testImage).stream().map(faceRect -> {
            final Mat img = convertToGray(new Mat(testImage, faceRect));

            IntPointer detectedPersonPtr = new IntPointer(1);
            DoublePointer confidencePtr = new DoublePointer(1);

            faceRecognizer.predict(img, detectedPersonPtr, confidencePtr);

            final String detectedPerson = faceRecognizer.getLabelInfo(detectedPersonPtr.get(0)).getString();
            final double detectionConfidence = confidencePtr.get(0);

            drawRectangleAndName(testImage, detectedPerson, faceRect);

            return new FaceRecognition.DetectedFaces(detectedPerson, detectionConfidence);
        }).toList();

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

            Mat img = convertToGray((Mat)data[0]);

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

        return faceRecognizer;
    }

    private static Mat convertToGray(Mat testImage) {
        Mat target = new Mat();
        cvtColor(testImage, target, COLOR_RGB2GRAY);
        return target;
    }

    private void drawRectangleAndName(Mat img, String text, Rect rect) {
        int textX = rect.x(); // or adjust for centering
        int textY = rect.y()+rect.height()+25; // offset to create space below rectangle.
        int fontFace = FONT_HERSHEY_SIMPLEX;
        double fontScale = 1.0;
        Scalar color = new Scalar(76, 175, 80, 1);
        int thicknessText = 2;
        int lineType = LINE_8;
        rectangle(img, rect, color, thicknessText, lineType, 0);
        putText(img, text, new Point(textX, textY), fontFace, fontScale, color, thicknessText, lineType, false);
    }
}
