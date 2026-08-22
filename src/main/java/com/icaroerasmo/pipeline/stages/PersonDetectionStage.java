package com.icaroerasmo.pipeline.stages;

import com.icaroerasmo.detectors.person.PersonDetector;
import com.icaroerasmo.pipeline.FrameContext;
import com.icaroerasmo.pipeline.FrameStage;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.bytedeco.opencv.opencv_core.Rect;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * STEP 1: First detect if there are any people in the frame.
 * When no people are detected the frame is dropped (processing marked complete).
 */
@Log4j2
@Component
@RequiredArgsConstructor
public class PersonDetectionStage implements FrameStage {

    private final PersonDetector personDetector;

    @Override
    public void process(FrameContext ctx) {
        // STEP 1: First detect if there are any people in the frame
        List<Rect> detectedPeople = personDetector.detect(ctx.getFrame());
        ctx.setDetectedPeople(detectedPeople);

        if (detectedPeople.isEmpty()) {
            // No people detected at all - skip this frame
            ctx.markProcessingComplete();
            return;
        }

        log.debug("Camera '{}': Detected {} person(s) in frame", ctx.getCameraName(), detectedPeople.size());
    }
}
