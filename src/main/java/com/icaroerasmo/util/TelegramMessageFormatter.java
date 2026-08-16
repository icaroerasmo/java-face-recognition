package com.icaroerasmo.util;

import com.icaroerasmo.enums.MessagesEnum;

public final class TelegramMessageFormatter {

    private TelegramMessageFormatter() {}

    private static String escapeHtml(String s) {
        if (s == null) return "";
        return s.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\"", "&quot;")
                .replace("'", "&#39;");
    }

    public static String format(MessagesEnum messageType, String translatedText) {
        String content = translatedText == null ? "" : translatedText;

        String prefix = "";
        switch (messageType) {
            case CAM_RECONNECTING:
                prefix = "🔄 ";
                break;
            case CAM_CONNECTED:
                prefix = "✅ ";
                break;
            case CAM_HIBERNATING:
                prefix = "😴 ";
                break;
            case CAM_HIBERNATE_COMPLETE:
                prefix = "⏰ ";
                break;
            default:
                prefix = "";
                break;
        }

        String escaped = escapeHtml(content);
        return prefix + escaped;
    }
}
