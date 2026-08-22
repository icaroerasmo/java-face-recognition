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
 * Publishes the movement alert (overlay event debounced, Telegram text throttled
 * and gated by {@code movement.notifyTelegram}). Only reached when no people and no
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

            // Telegram text (throttled per camera + notifyTelegram flag).
            if (movementProperties.isNotifyTelegram()
                    && movementAlertPolicy.shouldSendTelegram(cameraName, now, movementProperties.getTelegramThrottleMs())) {
                telegramPublisherService.sendMovementAlert(cameraName);
            }
        } catch (Exception e) {
            log.error("Failed to publish movement alert for camera '{}': {}", cameraName, e.getMessage(), e);
        }
    }
}
