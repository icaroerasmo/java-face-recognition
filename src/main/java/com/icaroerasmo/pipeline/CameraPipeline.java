package com.icaroerasmo.pipeline;

import com.icaroerasmo.detectors.movement.MovementResultStore;
import com.icaroerasmo.pipeline.stages.FaceRecognitionStage;
import com.icaroerasmo.pipeline.stages.MovementAlertStage;
import com.icaroerasmo.pipeline.stages.PeopleTrackingStage;
import com.icaroerasmo.pipeline.stages.PersonDetectionStage;
import com.icaroerasmo.pipeline.stages.PetDetectionStage;
import com.icaroerasmo.properties.ObjectDetectionProperties;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.springframework.stereotype.Component;

/**
 * Per-frame flow with alert priority <b>person &gt; pet &gt; movement</b>:
 * <ol>
 *   <li>Compute movement (only when {@code detection.movement.enabled}) - state is
 *       always kept fresh, alerts are gated by the priority below.</li>
 *   <li>Detect people - always, this is the core feature.</li>
 *   <li>Detect pets (only when {@code detection.pet.enabled} and no people present).</li>
 *   <li>Priority:
 *     <ul>
 *       <li>People present -&gt; run the existing face-recognition + people-tracking
 *           path unchanged; movement and pet alerts are suppressed.</li>
 *       <li>Else pets present -&gt; pet alert (Telegram PHOTO + overlay event).</li>
 *       <li>Else movement present -&gt; movement alert (Telegram text + overlay event).</li>
 *       <li>Else nothing.</li>
 *     </ul>
 *   </li>
 * </ol>
 *
 * <p>Movement and pet detection are independent of the master
 * {@code object-detection.enabled} flag (which gates face recognition only); they
 * are controlled by their own {@code enabled} flags.
 */
@Log4j2
@Component
@RequiredArgsConstructor
public class CameraPipeline {

    private final MovementResultStore movementResultStore;
    private final PersonDetectionStage personDetectionStage;
    private final PetDetectionStage petDetectionStage;
    private final MovementAlertStage movementAlertStage;
    private final FaceRecognitionStage faceRecognitionStage;
    private final PeopleTrackingStage peopleTrackingStage;
    private final ObjectDetectionProperties objectDetectionProperties;

    public void process(FrameContext ctx) {
        boolean movementEnabled = objectDetectionProperties.getDetection().getMovement().isEnabled();
        boolean petEnabled = objectDetectionProperties.getDetection().getPet().isEnabled();

        // 1. Movement state: read the verdict produced at a high rate by the producer
        //    (MovementResultStore), so movement alerts are not delayed by the DNN.
        if (movementEnabled) {
            try {
                ctx.setMovementDetected(movementResultStore.isMovementRecent(ctx.getCameraName()));
            } catch (Exception e) {
                log.error("Movement detection failed for camera '{}': {}", ctx.getCameraName(), e.getMessage(), e);
                ctx.setMovementDetected(false);
            }
        }

        // 2. People detection - always, core feature
        personDetectionStage.process(ctx);

        if (ctx.getDetectedPeople().isEmpty()) {
            // 3/4. No people -> pet alert, else movement alert (priority person > pet > movement)
            if (petEnabled) {
                try {
                    if (petDetectionStage.publishPetAlert(ctx)) {
                        ctx.markProcessingComplete();
                        return;
                    }
                } catch (Exception e) {
                    log.error("Pet detection stage failed for camera '{}': {}", ctx.getCameraName(), e.getMessage(), e);
                }
            }
            if (ctx.isMovementDetected()) {
                try {
                    movementAlertStage.publishMovementAlert(ctx);
                } catch (Exception e) {
                    log.error("Movement alert stage failed for camera '{}': {}", ctx.getCameraName(), e.getMessage(), e);
                }
            }
            ctx.markProcessingComplete();
            return;
        }

        // People present -> face recognition + people tracking (unchanged behavior)
        faceRecognitionStage.process(ctx);
        if (ctx.isProcessingComplete()) {
            return;
        }
        peopleTrackingStage.process(ctx);
    }
}
