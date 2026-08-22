package com.icaroerasmo.messaging;

import com.icaroerasmo.enums.MessagesEnum;
import lombok.extern.log4j.Log4j2;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

@Log4j2
@Service
public class DetectionEventPublisher {

    private static final String EXCHANGE = "detection.exchange";
    private static final String ROUTING_KEY = "detection.events";
    private static final long PRESENCE_DEBOUNCE_MS = 2000;

    @Autowired
    private RabbitTemplate rabbitTemplate;

    private final Map<String, Long> lastPresencePublish = new ConcurrentHashMap<>();

    public void publish(String cameraName, MessagesEnum template, List<String> args) {
        DetectionEvent event = new DetectionEvent(
                UUID.randomUUID().toString(),
                cameraName,
                template.name(),
                args);

        try {
            rabbitTemplate.convertAndSend(EXCHANGE, ROUTING_KEY, event);
            log.debug("Published detection event: camera={}, template={}", cameraName, template);
        } catch (Exception e) {
            log.error("Error publishing detection event to RabbitMQ: {}", e.getMessage());
        }
    }

    /**
     * Publishes a low-latency "person present" event, debounced per camera,
     * so the live-stream overlay appears as soon as a person is detected
     * (without waiting for the multi-frame tracking verdict).
     */
    public void publishPresence(String cameraName, String personName) {
        long now = System.currentTimeMillis();
        Long last = lastPresencePublish.get(cameraName);
        if (last != null && now - last < PRESENCE_DEBOUNCE_MS) {
            return;
        }
        lastPresencePublish.put(cameraName, now);

        boolean known = personName != null && !personName.isBlank()
                && !"Unknown".equalsIgnoreCase(personName);
        MessagesEnum template = known ? MessagesEnum.PERSON_DETECTED_KNOWN : MessagesEnum.PERSON_DETECTED_UNKNOWN;
        List<String> args = known ? List.of(personName) : List.of();
        publish(cameraName, template, args);
    }
}
