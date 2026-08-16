package com.icaroerasmo.service;


import com.icaroerasmo.enums.MessagesEnum;
import com.icaroerasmo.properties.TelegramProperties;
import com.pengrad.telegrambot.TelegramBot;
import com.pengrad.telegrambot.model.request.ParseMode;
import com.pengrad.telegrambot.request.SendAnimation;
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

    private static final String APP_TAG = "[Face Recognition] ";

    private final TelegramProperties telegramProperties;
    private final TelegramBot telegramBot;
    private final TranslationService translationService;

    /**
     * Sends face detection images to Telegram:
     * - Recognized people: Image with person's name in caption
     * - Unknown people: Image with "Unknown Person" in caption
     */
    public void publishDetection(byte[] imageBytes, Map<String, Double> detectedPeopleWithScores, String cameraName, int identityFrameCount, int totalTrackedFrames) {

        try {
            log.info("publishDetection called: cameraName={}, imageBytes={}, detectedPeople={}, identityFrameCount={}, totalTrackedFrames={}",
                cameraName, (imageBytes != null ? imageBytes.length : 0), detectedPeopleWithScores.keySet(), identityFrameCount, totalTrackedFrames);

            if (telegramBot == null) {
                log.error("FATAL: Telegram bot is not initialized!");
                throw new RuntimeException("Telegram bot is not initialized");
            }

            if (imageBytes == null || imageBytes.length == 0) {
                log.error("FATAL: Image bytes is null or empty!");
                throw new RuntimeException("Image bytes is null or empty");
            }

            // Build caption with detected people info
            String caption = buildCaption(detectedPeopleWithScores, cameraName, identityFrameCount, totalTrackedFrames);

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
     * Builds a caption for the Telegram message with detected people information.
     * Shows count summary: "✓ 1 known (Icaro), 2 unknown people"
     */
    private String buildCaption(Map<String, Double> detectedPeopleWithScores, String cameraName, int identityFrameCount, int totalTrackedFrames) {
        StringBuilder caption = new StringBuilder();

        caption.append("<b>").append(translationService.getMessage(MessagesEnum.DETECTION_HEADER, cameraName)).append("</b>\n");

        // Find and display the lowest distance (best match) from known people only
        double lowestDistance = detectedPeopleWithScores.entrySet().stream()
            .filter(e -> !"Unknown".equalsIgnoreCase(e.getKey()))
            .mapToDouble(Map.Entry::getValue)
            .min()
            .orElse(100.0);
        caption.append("<b>").append(translationService.getMessage(MessagesEnum.DETECTION_BEST_MATCH, String.format("%.2f", lowestDistance))).append("</b>\n");
        caption.append("<b>").append(translationService.getMessage(MessagesEnum.DETECTION_FRAMES_IDENTIFIED, identityFrameCount)).append("</b>\n");
        caption.append("<b>").append(translationService.getMessage(MessagesEnum.DETECTION_FRAMES_TRACKED, totalTrackedFrames)).append("</b>\n\n");

        // Count known and unknown people
        int unknownCount = 0;
        int knownCount = 0;
        StringBuilder knownNames = new StringBuilder();

        for (Map.Entry<String, Double> entry : detectedPeopleWithScores.entrySet()) {
            String personName = entry.getKey();
            double value = entry.getValue();

            if ("Unknown".equalsIgnoreCase(personName)) {
                // For Unknown entries, the value is the count of unknown people
                unknownCount += (int) Math.round(value);
            } else {
                knownCount++;
                if (knownNames.length() > 0) knownNames.append(", ");
                knownNames.append(personName);
            }
        }

        caption.append("<b>Detected:</b>\n");
        if (knownCount > 0) {
            caption.append("✓ ").append(translationService.getMessage(MessagesEnum.DETECTION_KNOWN, knownCount, knownNames.toString())).append("\n");
        }
        if (unknownCount > 0) {
            caption.append("🔍 ").append(translationService.getMessage(MessagesEnum.DETECTION_UNKNOWN, unknownCount)).append("\n");
        }
        if (knownNames.length() == 0 && unknownCount == 0) {
            caption.append(translationService.getMessage(MessagesEnum.DETECTION_NONE)).append("\n");
        }

        caption.append("\n").append(translationService.getMessage(MessagesEnum.DETECTION_TIME, java.time.LocalDateTime.now().format(java.time.format.DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"))));

        return caption.toString();
    }

    /**
     * Sends a translated text message to Telegram (for status notifications)
     */
    public void sendTranslatedMessage(MessagesEnum message, Object... params) {
        try {
            if (telegramBot == null) {
                log.error("Telegram bot is not initialized. Cannot send message: {}", message);
                return;
            }

            String translated = translationService.getMessage(message, params);
            String tagged = APP_TAG + translated;
            SendMessage sendMessage = new SendMessage(telegramProperties.getChatId(), tagged);
            SendResponse response = telegramBot.execute(sendMessage);

            if (response.isOk()) {
                log.debug("Successfully sent translated message to Telegram: {}", tagged);
            } else {
                log.warn("Failed to send translated message to Telegram: {}", response.description());
            }

        } catch (Exception e) {
            log.warn("Error sending translated message to Telegram: {}", e.getMessage());
        }
    }

    /**
     * Sends a raw text message to Telegram (for status notifications)
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

    /**
     * Sends a GIF animation to Telegram
     */
    public void sendAnimation(byte[] gifBytes, String caption, String cameraName) {
        try {
            if (telegramBot == null) {
                log.error("Telegram bot is not initialized. Cannot send animation");
                return;
            }

            if (gifBytes == null || gifBytes.length == 0) {
                log.error("Cannot send animation: GIF bytes is null or empty");
                return;
            }

            log.info("Sending GIF animation to Telegram: chatId={}, size={} bytes, caption={}",
                telegramProperties.getChatId(), gifBytes.length, caption);

            SendAnimation sendAnimation = new SendAnimation(telegramProperties.getChatId(), gifBytes)
                    .caption(caption)
                    .parseMode(ParseMode.HTML);

            SendResponse response = telegramBot.execute(sendAnimation);

            if (response.isOk()) {
                log.info("✅ Successfully sent GIF animation to Telegram for camera '{}'", cameraName);
            } else {
                log.error("Failed to send GIF to Telegram: {} (error code: {})",
                    response.description(), response.errorCode());
            }

        } catch (Exception e) {
            log.error("❌ Failed to send GIF animation to Telegram: {}", e.getMessage(), e);
        }
    }
}
