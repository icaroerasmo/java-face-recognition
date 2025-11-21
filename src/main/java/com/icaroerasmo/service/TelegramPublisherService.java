package com.icaroerasmo.service;


import com.icaroerasmo.properties.TelegramProperties;
import com.pengrad.telegrambot.TelegramBot;
import com.pengrad.telegrambot.model.request.ParseMode;
import com.pengrad.telegrambot.request.SendPhoto;
import com.pengrad.telegrambot.response.SendResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.springframework.stereotype.Service;

import java.util.Map;

@Log4j2
@Service
@RequiredArgsConstructor
public class TelegramPublisherService {

    private final TelegramProperties telegramProperties;
    private final TelegramBot telegramBot;

    /**
     * Sends face detection images to Telegram:
     * - Recognized people: Image with person's name in caption
     * - Unknown people: Image with "Unknown Person" in caption
     */
    public void publishDetection(byte[] imageBytes, Map<String, Double> detectedPeopleWithScores, String cameraName) {

        try {
            if (telegramBot == null) {
                log.error("FATAL: Telegram bot is not initialized. Application will terminate.");
                System.exit(1);
            }

            // Build caption with detected people info
            String caption = buildCaption(detectedPeopleWithScores, cameraName);

            // Send image to Telegram
            SendPhoto sendPhoto = new SendPhoto(telegramProperties.getChatId(), imageBytes)
                    .caption(caption)
                    .parseMode(ParseMode.HTML);

            SendResponse response = telegramBot.execute(sendPhoto);

            if (response.isOk()) {
                log.info("Successfully sent detection image to Telegram for camera '{}'. Caption: {}", cameraName, caption);
            } else {
                String errorMsg = "Failed to send image to Telegram: " + response.description();
                log.error(errorMsg);
                throw new RuntimeException(errorMsg);
            }

        } catch (Exception e) {
            log.error("FATAL: Failed to publish detection to Telegram. Application will terminate.", e);
            System.exit(1);
        }
    }

    /**
     * Builds a caption for the Telegram message with detected people information
     */
    private String buildCaption(Map<String, Double> detectedPeopleWithScores, String cameraName) {
        StringBuilder caption = new StringBuilder();
        caption.append("<b>Camera: ").append(cameraName).append("</b>\n");
        caption.append("<b>Detected:</b>\n");

        for (Map.Entry<String, Double> entry : detectedPeopleWithScores.entrySet()) {
            String personName = entry.getKey();
            double confidence = entry.getValue();
            double calculatedConfidence = Math.abs(1.0 - confidence);

            final String fomattedPercentage = String.format("%.2f", calculatedConfidence);

            if ("Unknown".equalsIgnoreCase(personName)) {
                caption.append("🔍 Unknown Person (").append(fomattedPercentage).append("%)\n");
            } else {
                caption.append("✓ ").append(personName).append(" (").append(fomattedPercentage).append("%)\n");
            }
        }

        caption.append("\nTime: ").append(java.time.LocalDateTime.now().format(java.time.format.DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")));

        return caption.toString();
    }
}
