package com.icaroerasmo.pipeline.stages;

import com.icaroerasmo.detectors.person.services.FaceRecognitionRuntime;
import com.icaroerasmo.detectors.person.services.FaceRecognitionService;
import com.icaroerasmo.detectors.person.services.FaceRecognizerHolderService;
import com.icaroerasmo.model.FaceRecognition;
import com.icaroerasmo.pipeline.FrameContext;
import com.icaroerasmo.pipeline.FrameStage;
import com.icaroerasmo.properties.ObjectDetectionProperties;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.bytedeco.opencv.opencv_core.Mat;
import org.springframework.stereotype.Component;

/**
 * STEP 2: Try to recognize faces in the frame, only when face recognition is
 * enabled. When no recognizer is initialized yet, the frame is skipped entirely.
 * When face recognition is disabled, the faceRecognition result is left null so
 * the tracking stage treats every detected person as unknown.
 */
@Log4j2
@Component
@RequiredArgsConstructor
public class FaceRecognitionStage implements FrameStage {

    private final FaceRecognitionService faceRecognitionService;
    private final FaceRecognizerHolderService faceRecognizerHolderService;
    private final ObjectDetectionProperties objectDetectionProperties;

    @Override
    public void process(FrameContext ctx) {
        if (!objectDetectionProperties.getEnabled()) {
            return;
        }

        // STEP 2: Try to recognize faces in the frame
        FaceRecognition faceRecognition = getFaceRecognition(ctx.getFrame());
        ctx.setFaceRecognition(faceRecognition);

        if (faceRecognition == null) {
            ctx.markProcessingComplete();
        }
    }

    private FaceRecognition getFaceRecognition(Mat img) {
        // Get the current recognizer from the holder (thread-safe)
        FaceRecognitionRuntime currentRecognizer = faceRecognizerHolderService.get();

        if (currentRecognizer == null) {
            log.warn("FaceRecognizer not initialized yet, skipping frame");
            return null;
        }

        // STEP 2: Try to recognize faces in the frame
        return faceRecognitionService.test(currentRecognizer, img);
    }
}
