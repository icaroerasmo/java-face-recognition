package com.icaroerasmo.detectors.person.services;

import lombok.Getter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_objdetect.FaceRecognizerSF;

import java.util.List;
import java.util.Map;

@Getter
public class FaceRecognitionRuntime implements AutoCloseable {

    private final FaceRecognizerSF recognizer;
    private final Map<String, List<Mat>> galleryFeatures;

    public FaceRecognitionRuntime(FaceRecognizerSF recognizer, Map<String, List<Mat>> galleryFeatures) {
        this.recognizer = recognizer;
        this.galleryFeatures = galleryFeatures;
    }

    @Override
    public void close() {
        galleryFeatures.values().forEach(features ->
            features.forEach(feature -> {
                if (feature != null) {
                    feature.release();
                }
            })
        );

        if (recognizer != null) {
            try {
                recognizer.close();
            } catch (Exception ignore) {
            }
        }
    }
}
