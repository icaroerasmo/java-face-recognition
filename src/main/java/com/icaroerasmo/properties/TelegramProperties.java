package com.icaroerasmo.properties;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

@Data
@Component
@ConfigurationProperties(prefix = "face-recognition.telegram")
public class TelegramProperties {
    // Bot token from Telegram BotFather
    private String botToken;
    // Telegram chat ID to send notifications to
    private String chatId;
    // Animation FPS for Telegram MP4/GIF previews
    private int gifFps = 10;
}
