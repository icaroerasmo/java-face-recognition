package com.icaroerasmo.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.icaroerasmo.properties.MqttProperties;
import lombok.extern.log4j.Log4j2;
import org.eclipse.paho.client.mqttv3.*;
import org.eclipse.paho.client.mqttv3.persist.MemoryPersistence;
import org.springframework.stereotype.Service;

import jakarta.annotation.PreDestroy;
import java.util.Base64;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Log4j2
@Service
public class MqttPublisherService {

    private final MqttProperties mqttProperties;
    private final ObjectMapper objectMapper;
    private MqttClient mqttClient;

    public MqttPublisherService(MqttProperties mqttProperties) {
        this.mqttProperties = mqttProperties;
        this.objectMapper = new ObjectMapper();

        if (mqttProperties.isEnabled()) {
            initializeMqttClient();
        } else {
            log.info("MQTT publishing is disabled");
        }
    }

    private void initializeMqttClient() {
        try {
            mqttClient = new MqttClient(
                mqttProperties.getBroker(),
                mqttProperties.getClientId(),
                new MemoryPersistence()
            );

            MqttConnectOptions options = new MqttConnectOptions();
            options.setCleanSession(true);
            options.setConnectionTimeout(mqttProperties.getConnectionTimeout());
            options.setKeepAliveInterval(mqttProperties.getKeepAliveInterval());
            options.setAutomaticReconnect(true);

            if (mqttProperties.getUsername() != null && !mqttProperties.getUsername().isEmpty()) {
                options.setUserName(mqttProperties.getUsername());
            }
            if (mqttProperties.getPassword() != null && !mqttProperties.getPassword().isEmpty()) {
                options.setPassword(mqttProperties.getPassword().toCharArray());
            }

            mqttClient.connect(options);
            log.info("MQTT client connected to broker: {}", mqttProperties.getBroker());

        } catch (MqttException e) {
            log.error("Failed to initialize MQTT client", e);
        }
    }

    /**
     * Publishes face detection in double-take compatible format
     * Topic: {topicPrefix}/matches
     * Payload: JSON with detected faces and image
     */
    public void publishDetection(byte[] imageBytes, Map<String, Double> detectedPeopleWithScores, String cameraName) {
        if (!mqttProperties.isEnabled() || mqttClient == null || !mqttClient.isConnected()) {
            log.debug("MQTT client not available, skipping publish");
            return;
        }

        try {
            // Build double-take compatible payload
            Map<String, Object> payload = new HashMap<>();
            payload.put("camera", cameraName);
            payload.put("type", "match");

            // Encode image as base64
            String base64Image = Base64.getEncoder().encodeToString(imageBytes);
            payload.put("image", base64Image);

            // Add matches array (detected people with calculated confidence)
            List<Map<String, Object>> matches = detectedPeopleWithScores.entrySet().stream()
                .map(entry -> {
                    Map<String, Object> match = new HashMap<>();
                    match.put("name", entry.getKey());
                    // Calculate confidence: confidence = |1 - score|
                    double confidence = Math.abs(1.0 - entry.getValue());
                    match.put("confidence", confidence);
                    return match;
                })
                .toList();
            payload.put("matches", matches);

            // Add timestamp
            payload.put("timestamp", System.currentTimeMillis());

            // Convert to JSON
            String jsonPayload = objectMapper.writeValueAsString(payload);

            // Publish to MQTT
            String topic = mqttProperties.getTopicPrefix() + "/matches";
            MqttMessage message = new MqttMessage(jsonPayload.getBytes());
            message.setQos(mqttProperties.getQos());
            message.setRetained(false);

            mqttClient.publish(topic, message);
            log.info("Published detection to MQTT topic '{}' for people: {}", topic, detectedPeopleWithScores.keySet());

        } catch (Exception e) {
            log.error("Failed to publish detection to MQTT", e);
        }
    }

    /**
     * Publishes individual person detection
     * Topic: {topicPrefix}/person/{personName}
     */
    public void publishPersonDetection(byte[] imageBytes, String personName, double score) {
        if (!mqttProperties.isEnabled() || mqttClient == null || !mqttClient.isConnected()) {
            return;
        }

        try {
            // Calculate confidence: confidence = |1 - score|
            double calculatedConfidence = Math.abs(1.0 - score);

            Map<String, Object> payload = new HashMap<>();
            payload.put("name", personName);
            payload.put("confidence", calculatedConfidence);
            payload.put("timestamp", System.currentTimeMillis());

            String base64Image = Base64.getEncoder().encodeToString(imageBytes);
            payload.put("image", base64Image);

            String jsonPayload = objectMapper.writeValueAsString(payload);

            String topic = mqttProperties.getTopicPrefix() + "/person/" + personName.toLowerCase();
            MqttMessage message = new MqttMessage(jsonPayload.getBytes());
            message.setQos(mqttProperties.getQos());
            message.setRetained(false);

            mqttClient.publish(topic, message);
            log.debug("Published person detection to MQTT topic '{}' with confidence {}", topic, calculatedConfidence);

        } catch (Exception e) {
            log.error("Failed to publish person detection to MQTT", e);
        }
    }

    @PreDestroy
    public void disconnect() {
        if (mqttClient != null && mqttClient.isConnected()) {
            try {
                mqttClient.disconnect();
                mqttClient.close();
                log.info("MQTT client disconnected");
            } catch (MqttException e) {
                log.error("Error disconnecting MQTT client", e);
            }
        }
    }
}

