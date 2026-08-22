package com.icaroerasmo.enums;

public enum MessagesEnum {
    // Camera lifecycle
    CAM_RECONNECTING,
    CAM_CONNECTED,
    CAM_HIBERNATING,
    CAM_HIBERNATE_COMPLETE,

    // Detection events (published to live-transmission overlay via detection.exchange)
    PERSON_DETECTED
}
