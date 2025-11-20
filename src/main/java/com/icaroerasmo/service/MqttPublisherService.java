package com.icaroerasmo.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.icaroerasmo.properties.MqttProperties;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.eclipse.paho.client.mqttv3.*;
import org.eclipse.paho.client.mqttv3.persist.MemoryPersistence;
import org.springframework.stereotype.Service;

import java.util.Base64;
import java.util.HashMap;
import java.util.Map;

@Log4j2
@Service
@RequiredArgsConstructor
public class MqttPublisherService {

    // Hard-coded topic prefix constant
    private static final String TOPIC_PREFIX = "double-take";

    private final MqttProperties mqttProperties;
    private final ObjectMapper objectMapper = new ObjectMapper();


    public MqttClient connect() {
        final String clientId = "mqttListener";
        final String connectionString = "%s://%s:%s".
                formatted(mqttProperties.getProtocol().getProtocolShort(),
                        mqttProperties.getHost(), mqttProperties.getPort());

        try {
            // Use MemoryPersistence to avoid file system access issues
            MqttClient mqttClient = new MqttClient(connectionString, clientId, new MemoryPersistence());

            MqttConnectOptions options = new MqttConnectOptions();
            options.setUserName(mqttProperties.getUsername());
            options.setPassword(mqttProperties.getPassword().toCharArray());
            options.setAutomaticReconnect(mqttProperties.getAutomaticReconnect());
            options.setCleanSession(mqttProperties.getCleanSession());
            options.setConnectionTimeout(mqttProperties.getConnectionTimeout());

            mqttClient.connect(options);

            log.info("MQTT client connected to broker: {}:{}", mqttProperties.getHost(), mqttProperties.getPort());

            return mqttClient;

        } catch (MqttException e) {
            log.error("Error connecting to MQTT broker. Exiting...", e);
        }
        System.exit(1);
        throw new IllegalStateException("Failed to connect to MQTT broker");
    }

    /**
     * Publishes face detection to appropriate MQTT topics:
     * - Recognized persons: {topicPrefix}/{cameraName}/{personName}
     * - Unknown persons: {topicPrefix}/person/unknown ONLY
     * Payload: JSON with person name, confidence and image
     */
    public void publishDetection(byte[] imageBytes, Map<String, Double> detectedPeopleWithScores, String cameraName) {

        try {

            MqttClient mqttClient = connect();

            // Fail fast: MQTT connection is critical for operation
            if (!mqttClient.isConnected()) {
                String errorMsg = "MQTT client is not connected. Application cannot continue without MQTT connectivity.";
                log.error(errorMsg);
                throw new IllegalStateException(errorMsg);
            }

            String base64Image = Base64.getEncoder().encodeToString(imageBytes);

            // Publish each detected person to appropriate topic based on whether they are recognized or unknown
            for (Map.Entry<String, Double> entry : detectedPeopleWithScores.entrySet()) {
                String personName = entry.getKey();
                double confidence = entry.getValue();

                // Calculate confidence: confidence = |1 - score|
                double calculatedConfidence = Math.abs(1.0 - confidence);

                Map<String, Object> payload = new HashMap<>();
                payload.put("name", personName);
                payload.put("confidence", calculatedConfidence);
                payload.put("camera", cameraName);
                payload.put("timestamp", System.currentTimeMillis());
                payload.put("image", base64Image);

                String jsonPayload = objectMapper.writeValueAsString(payload);
                MqttMessage message = new MqttMessage(jsonPayload.getBytes());
                message.setRetained(false);

                // Check if this is an Unknown detection or a recognized person
                if ("Unknown".equalsIgnoreCase(personName)) {
                    // Unknown persons go ONLY to person/unknown topic
                    String unknownTopic = TOPIC_PREFIX + "/person/unknown";
                    mqttClient.publish(unknownTopic, message);
                    log.info("Published unknown person detection to MQTT topic '{}' with confidence {}",
                        unknownTopic, calculatedConfidence);
                } else {
                    // Recognized persons go to camera-specific topic
                    String cameraTopicName = cameraName.toLowerCase().replaceAll("[^a-z0-9-_]", "_");
                    String personTopicName = personName.toLowerCase().replaceAll("[^a-z0-9-_]", "_");
                    String cameraTopic = TOPIC_PREFIX + "/" + cameraTopicName + "/" + personTopicName;

                    mqttClient.publish(cameraTopic, message);
                    log.info("Published detection to MQTT topic '{}' for person '{}' with confidence {}",
                        cameraTopic, personName, calculatedConfidence);
                }
            }

        } catch (Exception e) {
            log.error("FATAL: Failed to publish detection to MQTT. Application will terminate.", e);
            System.exit(1);
        }
    }

//    @PreDestroy
//    public void disconnect() {
//        if (mqttClient != null && mqttClient.isConnected()) {
//            try {
//                mqttClient.disconnect();
//                mqttClient.close();
//                log.info("MQTT client disconnected");
//            } catch (MqttException e) {
//                log.error("Error disconnecting MQTT client", e);
//            }
//        }
//    }
}
