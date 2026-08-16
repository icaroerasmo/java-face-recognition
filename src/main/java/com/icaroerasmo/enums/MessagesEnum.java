package com.icaroerasmo.enums;

public enum MessagesEnum {
    // Camera lifecycle
    CAM_RECONNECTING,
    CAM_CONNECTED,
    CAM_HIBERNATING,
    CAM_HIBERNATE_COMPLETE,

    // Detection notifications
    DETECTION_HEADER,
    DETECTION_BEST_MATCH,
    DETECTION_FRAMES_IDENTIFIED,
    DETECTION_FRAMES_TRACKED,
    DETECTION_KNOWN,
    DETECTION_UNKNOWN,
    DETECTION_NONE,
    DETECTION_TIME,

    // GIF notification
    GIF_HEADER,
    GIF_CAMERA,
    GIF_FRAMES,
    GIF_DURATION
}
