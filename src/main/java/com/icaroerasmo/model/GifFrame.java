package com.icaroerasmo.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.bytedeco.opencv.opencv_core.Mat;

import java.time.Instant;

@Data
@AllArgsConstructor
public class GifFrame {
    private Instant timestamp;
    private Mat frame;
}
