package com.icaroerasmo.pipeline.stages;

import com.icaroerasmo.detectors.movement.MovementAlertPolicy;
import com.icaroerasmo.messaging.DetectionEventPublisher;
import com.icaroerasmo.pipeline.FrameContext;
import com.icaroerasmo.properties.MovementDetectionProperties;
import com.icaroerasmo.properties.ObjectDetectionProperties;
import com.icaroerasmo.service.TelegramPublisherService;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Component;

/**
 * Publishes the movement alert (overlay event debounced, notification text throttled
 * and gated by {@code movement.notify}). Only reached when no people and no
 * pets were detected in the frame.
 */
@Log4j2
@Component
@RequiredArgsConstructor
public class MovementAlertStage {

    private final DetectionEventPublisher detectionEventPublisher;
    private final TelegramPublisherService telegramPublisherService;
    @Qualifier("movementAlertPolicy")
    private final MovementAlertPolicy movementAlertPolicy;
    private final ObjectDetectionProperties objectDetectionProperties;

    public void publishMovementAlert(FrameContext ctx) {
        MovementDetectionProperties movementProperties = objectDetectionProperties.getDetection().getMovement();
        String cameraName = ctx.getCameraName();
        long now = System.currentTimeMillis();
        try {
            // Overlay detection event (debounced per camera).
            if (movementAlertPolicy.shouldPublish(cameraName, now, movementProperties.getDebounceMs())) {
                detectionEventPublisher.publishMovement(cameraName, movementProperties.getDebounceMs());
            }

            // Notification text (throttled per camera + notify flag).
            if (movementProperties.isNotify()
                    && movementAlertPolicy.shouldSend(cameraName, now, movementProperties.getThrottleMs())) {
                telegramPublisherService.sendMovementAlert(cameraName);
            }
        } catch (Exception e) {
            log.error("Failed to publish movement alert for camera '{}': {}", cameraName, e.getMessage(), e);
        }
    }
}
