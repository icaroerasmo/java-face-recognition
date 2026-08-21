package com.icaroerasmo.messaging;

import java.util.List;

public record NotificationMessage(
        String messageId,
        String sender,
        MediaType mediaType,
        String template,
        List<String> args,
        String rawHtml,
        String filename,
        byte[] payload,
        boolean appendNoLogs) {

    public enum MediaType { TEXT, PHOTO, ANIMATION, DOCUMENT }
}
