package com.icaroerasmo.processing;

import com.icaroerasmo.utils.MatUtil;
import lombok.RequiredArgsConstructor;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.springframework.stereotype.Service;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imencode;

/**
 * Centralizes JPEG encoding of frames and frame regions.
 * All native resources (BytePointer, region Mat) are released in {@code finally}
 * blocks, replacing the repeated inline encoding/cleanup blocks in the runner.
 */
@Service
@RequiredArgsConstructor
public class FrameEncodingService {

    private final MatUtil matUtil;

    public byte[] encodeJpeg(Mat frame) {
        BytePointer buf = null;
        BytePointer jpgExt = null;
        try {
            buf = new BytePointer();
            jpgExt = new BytePointer(".jpg");
            imencode(jpgExt, frame, buf);
            byte[] imageBytes = new byte[(int) buf.limit()];
            buf.get(imageBytes);
            return imageBytes;
        } finally {
            if (buf != null) {
                buf.deallocate();
            }
            if (jpgExt != null) {
                jpgExt.deallocate();
            }
        }
    }

    public byte[] encodeRegionJpeg(Mat frame, Rect region) {
        Mat regionMat = null;
        BytePointer buf = null;
        BytePointer jpgExt = null;
        try {
            regionMat = new Mat(frame, region);
            buf = new BytePointer();
            jpgExt = new BytePointer(".jpg");
            imencode(jpgExt, regionMat, buf);
            byte[] encodedRegion = new byte[(int) buf.limit()];
            buf.get(encodedRegion);
            return encodedRegion;
        } finally {
            if (buf != null) {
                buf.deallocate();
            }
            if (jpgExt != null) {
                jpgExt.deallocate();
            }
            if (regionMat != null) {
                matUtil.releaseResources(regionMat);
            }
        }
    }
}
