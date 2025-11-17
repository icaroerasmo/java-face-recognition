package com.icaroerasmo.properties;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

@Data
@Component
@ConfigurationProperties(prefix = "face-recognition.mqtt")
public class MqttProperties {

    private boolean enabled = true;
    private String broker = "tcp://localhost:1883";
    private String clientId = "face-recognition-service";
    private String username = "";
    private String password = "";
    private String topicPrefix = "double-take";
    private int qos = 0;
    private int connectionTimeout = 10;
    private int keepAliveInterval = 60;
}

