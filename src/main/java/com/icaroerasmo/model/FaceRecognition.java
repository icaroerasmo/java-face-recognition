package com.icaroerasmo.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.bytedeco.opencv.opencv_core.Rect;

import java.util.List;

@Data
@AllArgsConstructor
public class FaceRecognition {
    private List<DetectedFaces> faces;

    @Data
    @AllArgsConstructor
    public static class DetectedFaces {
        private String personName;
        private Double distance;
        private Rect faceRect;
    }
}
