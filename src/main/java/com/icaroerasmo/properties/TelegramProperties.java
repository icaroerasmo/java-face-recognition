package com.icaroerasmo.properties;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

@Data
@Component
@ConfigurationProperties(prefix = "object-detection.telegram")
public class TelegramProperties {
    // Animation FPS for Telegram MP4/GIF previews
    private int gifFps = 10;
    // Maximum number of frame images retained per tracked person for Telegram clips
    private int gifMaxFrames = 30;
}
