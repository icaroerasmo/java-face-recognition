package com.icaroerasmo.service;


import com.icaroerasmo.properties.TelegramProperties;
import com.pengrad.telegrambot.TelegramBot;
import com.pengrad.telegrambot.model.request.ParseMode;
import com.pengrad.telegrambot.request.SendMessage;
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
            log.info("publishDetection called: cameraName={}, imageBytes={}, detectedPeople={}",
                cameraName, (imageBytes != null ? imageBytes.length : 0), detectedPeopleWithScores.keySet());

            if (telegramBot == null) {
                log.error("FATAL: Telegram bot is not initialized!");
                throw new RuntimeException("Telegram bot is not initialized");
            }

            if (imageBytes == null || imageBytes.length == 0) {
                log.error("FATAL: Image bytes is null or empty!");
                throw new RuntimeException("Image bytes is null or empty");
            }

            // Build caption with detected people info
            String caption = buildCaption(detectedPeopleWithScores, cameraName);

            log.info("Sending photo to Telegram: chatId={}, caption={}", telegramProperties.getChatId(), caption);

            // Send image to Telegram
            SendPhoto sendPhoto = new SendPhoto(telegramProperties.getChatId(), imageBytes)
                    .caption(caption)
                    .parseMode(ParseMode.HTML);

            SendResponse response = telegramBot.execute(sendPhoto);

            if (response.isOk()) {
                log.info("✅ Successfully sent detection image to Telegram for camera '{}'. Caption: {}", cameraName, caption);
            } else {
                String errorMsg = "Failed to send image to Telegram: " + response.description() + " (error code: " + response.errorCode() + ")";
                log.error(errorMsg);
                throw new RuntimeException(errorMsg);
            }

        } catch (Exception e) {
            log.error("❌ Failed to publish detection to Telegram: {}", e.getMessage(), e);
            throw new RuntimeException("Failed to publish detection to Telegram", e);
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
            double distance = entry.getValue();

            final String formattedDistance = String.format("%.2f", distance);

            if ("Unknown".equalsIgnoreCase(personName)) {
                caption.append("🔍 Unknown Person (distance: ").append(formattedDistance).append(")\n");
            } else {
                caption.append("✓ ").append(personName).append(" (distance: ").append(formattedDistance).append(")\n");
            }
        }

        caption.append("\nTime: ").append(java.time.LocalDateTime.now().format(java.time.format.DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")));

        return caption.toString();
    }

    /**
     * Sends a text message to Telegram (for status notifications)
     */
    public void sendTextMessage(String message) {
        try {
            if (telegramBot == null) {
                log.error("Telegram bot is not initialized. Cannot send message: {}", message);
                return;
            }

            SendMessage sendMessage = new SendMessage(telegramProperties.getChatId(), message);
            SendResponse response = telegramBot.execute(sendMessage);

            if (response.isOk()) {
                log.debug("Successfully sent text message to Telegram: {}", message);
            } else {
                log.warn("Failed to send text message to Telegram: {}", response.description());
            }

        } catch (Exception e) {
            log.warn("Error sending text message to Telegram: {}", e.getMessage());
        }
    }
}
