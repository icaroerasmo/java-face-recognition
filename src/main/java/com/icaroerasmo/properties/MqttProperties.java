package com.icaroerasmo.properties;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.Getter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

@Data
@Component
@ConfigurationProperties(prefix = "face-recognition.mqtt")
public class MqttProperties {
    // localhost
    private String host = "localhost";
    // 1883
    private String port = "1883";
    // TCP or WEBSOCKETS
    private ProtocolEnum protocol = ProtocolEnum.TCP;
    // mqtt_user
    private String username = "mqtt_user";
    // password
    private String password = "password";
    // true
    private Boolean automaticReconnect = true;
    //true
    private Boolean cleanSession = true;
    // 10
    private Integer connectionTimeout = 10;

    @Getter
    @AllArgsConstructor
    public enum ProtocolEnum {
        TCP("tcp"), WEBSOCKETS("ws");
        private String protocolShort;
    }
}
