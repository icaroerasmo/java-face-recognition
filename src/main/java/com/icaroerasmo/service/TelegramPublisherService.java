package com.icaroerasmo.service;

import com.icaroerasmo.enums.MessagesEnum;
import com.icaroerasmo.messaging.NotificationMessage;
import com.icaroerasmo.messaging.NotificationPublisher;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.springframework.stereotype.Service;

import java.util.Map;

@Log4j2
@Service
@RequiredArgsConstructor
public class TelegramPublisherService {

    private final NotificationPublisher publisher;

    /**
     * Publishes face detection images via RabbitMQ.
     * Sends raw data (detected people + camera info) so the notifier can build the message.
     */
    public void publishDetection(byte[] imageBytes, Map<String, Double> detectedPeopleWithScores, String cameraName, int identityFrameCount, int totalTrackedFrames) {

        try {
            log.info("publishDetection called: cameraName={}, imageBytes={}, detectedPeople={}, identityFrameCount={}, totalTrackedFrames={}",
                cameraName, (imageBytes != null ? imageBytes.length : 0), detectedPeopleWithScores.keySet(), identityFrameCount, totalTrackedFrames);

            if (imageBytes == null || imageBytes.length == 0) {
                log.error("FATAL: Image bytes is null or empty!");
                throw new RuntimeException("Image bytes is null or empty");
            }

            log.info("Publishing detection photo to RabbitMQ for camera '{}'", cameraName);

            // Publish raw data so the notifier can build the caption
            NotificationMessage.CaptionSpec captionSpec = new NotificationMessage.CaptionSpec(
                    cameraName, detectedPeopleWithScores, identityFrameCount, totalTrackedFrames, null, null);
            publisher.publishPhoto(captionSpec, imageBytes);

        } catch (Exception e) {
            log.error("❌ Failed to publish detection to RabbitMQ: {}", e.getMessage(), e);
            throw new RuntimeException("Failed to publish detection to RabbitMQ", e);
        }
    }

    /**
     * Publishes a translated text message via RabbitMQ (for status notifications).
     * The template name and arguments are sent so the notifier can render the message.
     */
    public void sendTranslatedMessage(MessagesEnum message, Object... params) {
        try {
            log.debug("Publishing translated message to RabbitMQ: template={}", message.name());
            publisher.publishText(message, params);
        } catch (Exception e) {
            log.warn("Error publishing translated message to RabbitMQ: {}", e.getMessage());
        }
    }

    /**
     * Publishes a GIF animation via RabbitMQ
     */
    public void sendAnimation(byte[] gifBytes, String cameraName, int frameCount, double duration) {
        try {
            if (gifBytes == null || gifBytes.length == 0) {
                log.error("Cannot send animation: GIF bytes is null or empty");
                return;
            }

            log.info("Publishing GIF animation to RabbitMQ: size={} bytes, camera={}, frameCount={}, duration={}",
                gifBytes.length, cameraName, frameCount, duration);

            NotificationMessage.CaptionSpec captionSpec = new NotificationMessage.CaptionSpec(
                    cameraName, null, null, null, frameCount, duration);
            publisher.publishAnimation(captionSpec, gifBytes);

        } catch (Exception e) {
            log.error("❌ Failed to publish GIF animation to RabbitMQ: {}", e.getMessage(), e);
        }
    }
}
