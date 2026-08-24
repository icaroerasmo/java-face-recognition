package com.icaroerasmo.detectors;

import com.icaroerasmo.pipeline.FrameContext;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;

import java.util.List;

public interface IDetector {
    /**
     * Detect people in an frameCtx.
     * THREAD-SAFE: Synchronized to prevent concurrent access to the shared Net object.
     *
     * @param frameCtx Input frameCtx
     * @return List of rectangles representing detected people
     */
    List<Rect> detect(FrameContext frameCtx);
}
