package com.icaroerasmo.messaging;

import java.util.List;
import java.util.Map;

public record NotificationMessage(
        String messageId,
        String sender,
        MediaType mediaType,
        String template,
        List<String> args,
        String rawHtml,
        CaptionSpec caption,
        String filename,
        byte[] payload,
        boolean appendNoLogs,
        String sentAt) {

    public enum MediaType { TEXT, PHOTO, ANIMATION, DOCUMENT }

    public record CaptionSpec(
            String cameraName,
            Map<String, Double> detectedPeople,
            Integer identityFrameCount,
            Integer totalTrackedFrames,
            Integer frameCount,
            Double duration) {
    }
}
