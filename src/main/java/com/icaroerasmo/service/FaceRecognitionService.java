package com.icaroerasmo.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.springframework.stereotype.Service;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.MatVector;
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
import static org.bytedeco.opencv.global.opencv_imgcodecs.IMREAD_GRAYSCALE;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;

@Log4j2
@Service
@RequiredArgsConstructor
public class FaceRecognitionService {

    private static final Path DATASET = Paths.get("trained_dataset.xml");

    public FaceRecognizer load() {
        FaceRecognizer faceRecognizer = LBPHFaceRecognizer.create();
        faceRecognizer.read(DATASET.toString());
        return faceRecognizer;
    }

    public Object[] test(FaceRecognizer faceRecognizer, String testFIle) {
        Mat testImage = imread(testFIle, IMREAD_GRAYSCALE);
        IntPointer label = new IntPointer(1);
        DoublePointer confidence = new DoublePointer(1);
        faceRecognizer.predict(testImage, label, confidence);
        return new Object[] {label.get(0), confidence.get(0)};
    }

    public FaceRecognizer train(String root) throws IOException {
        Path rootFolder = Paths.get(root);
        Map<Path, String> fileList = Files.list(rootFolder).
                filter(file -> file.toFile().isDirectory()).
                flatMap(folder -> {
                    try {
                        var personName = folder.getName(folder.getNameCount()-1).toString();
                        return Files.list(folder).map(file -> Map.entry(file, personName));
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                }).collect(toMap(Map.Entry::getKey, Map.Entry::getValue));

        MatVector images = new MatVector(fileList.size());

        List<String> strLabels = fileList.values().stream().distinct().toList();

        Mat labels = new Mat(fileList.size(), 1, CV_32SC1);
        IntBuffer labelsBuf = labels.createBuffer();

        final AtomicInteger counter = new AtomicInteger();

        fileList.keySet().forEach((Path key) -> {

            File image = key.toFile();

            Mat img = imread(image.getAbsolutePath(), IMREAD_GRAYSCALE);

            images.put(counter.get(), img);

            int imgLabel = strLabels.indexOf(fileList.get(key));

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
}
