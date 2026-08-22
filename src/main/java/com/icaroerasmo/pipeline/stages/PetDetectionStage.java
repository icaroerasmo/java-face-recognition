package com.icaroerasmo.pipeline.stages;

import com.icaroerasmo.detectors.movement.MovementAlertPolicy;
import com.icaroerasmo.detectors.pet.PetDetection;
import com.icaroerasmo.detectors.pet.PetDetector;
import com.icaroerasmo.messaging.DetectionEventPublisher;
import com.icaroerasmo.pipeline.FrameContext;
import com.icaroerasmo.processing.FrameEncodingService;
import com.icaroerasmo.properties.ObjectDetectionProperties;
import com.icaroerasmo.properties.PetDetectionProperties;
import com.icaroerasmo.service.TelegramPublisherService;
import com.icaroerasmo.utils.MatUtil;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.bytedeco.opencv.opencv_core.Mat;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * Detects pets (when enabled) on frames with no detected people and publishes the
 * pet alert: overlay {@code PET_DETECTED} event (debounced) plus an annotated
 * Telegram photo (throttled).
 *
 * <p>The pet rects are owned by this stage and deallocated in {@code finally};
 * they are never transferred into the {@link FrameContext}.
 */
@Log4j2
@Component
@RequiredArgsConstructor
public class PetDetectionStage {

    private final PetDetector petDetector;
    private final MatUtil matUtil;
    private final FrameEncodingService frameEncodingService;
    private final TelegramPublisherService telegramPublisherService;
    private final DetectionEventPublisher detectionEventPublisher;
    @Qualifier("petAlertPolicy")
    private final MovementAlertPolicy petAlertPolicy;
    private final ObjectDetectionProperties objectDetectionProperties;

    /**
     * @return {@code true} when at least one pet was detected and an alert was
     *         published (in which case the movement alert must be suppressed).
     */
    public boolean publishPetAlert(FrameContext ctx) {
        PetDetectionProperties petProperties = objectDetectionProperties.getDetection().getPet();
        String cameraName = ctx.getCameraName();
        List<PetDetection> pets = petDetector.detect(ctx.getFrame());
        if (pets.isEmpty()) {
            return false;
        }

        long now = System.currentTimeMillis();
        try {
            // Overlay detection event (debounced per camera).
            if (petAlertPolicy.shouldPublish(cameraName, now, petProperties.getDebounceMs())) {
                detectionEventPublisher.publishPet(cameraName, petProperties.getDebounceMs());
            }

            // Telegram photo (throttled per camera).
            if (petAlertPolicy.shouldSendTelegram(cameraName, now, petProperties.getTelegramThrottleMs())) {
                Mat annotated = null;
                try {
                    annotated = ctx.getFrame().clone();
                    for (PetDetection pet : pets) {
                        matUtil.drawRectangleAndName(annotated, pet.label(), pet.rect());
                    }
                    byte[] jpeg = frameEncodingService.encodeJpeg(annotated);
                    telegramPublisherService.sendPetPhoto(jpeg, cameraName);
                } finally {
                    if (annotated != null) {
                        matUtil.releaseResources(annotated);
                    }
                }
            }
        } catch (Exception e) {
            log.error("Failed to publish pet alert for camera '{}': {}", cameraName, e.getMessage(), e);
        } finally {
            // Pet rects are owned by this stage; the context does not track them.
            for (PetDetection pet : pets) {
                if (pet.rect() != null) {
                    pet.rect().deallocate();
                }
            }
        }
        return true;
    }
}
