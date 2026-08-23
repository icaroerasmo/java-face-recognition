package com.icaroerasmo.messaging;

import com.icaroerasmo.enums.MessagesEnum;
import lombok.extern.log4j.Log4j2;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;

@Log4j2
@Service
public class DetectionEventPublisher {

    private static final String EXCHANGE = "detection.exchange";
    private static final String ROUTING_KEY = "detection.events";
    private static final long PRESENCE_DEBOUNCE_MS = 2000;

    @Autowired
    private RabbitTemplate rabbitTemplate;

    private final Map<String, Long> lastPresencePublish = new ConcurrentHashMap<>();
    private final Map<String, Long> lastMovementPublish = new ConcurrentHashMap<>();
    private final Map<String, Long> lastPetPublish = new ConcurrentHashMap<>();

    @Async
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
    @Async
    public void publishPresence(String cameraName) {
        publishDebounced(cameraName, PRESENCE_DEBOUNCE_MS, MessagesEnum.PERSON_DETECTED, lastPresencePublish);
    }

    /**
     * Publishes a {@code MOVEMENT_DETECTED} overlay event, atomically debounced per
     * camera with the given window.
     */
    @Async
    public void publishMovement(String cameraName, long debounceMs) {
        publishDebounced(cameraName, debounceMs, MessagesEnum.MOVEMENT_DETECTED, lastMovementPublish);
    }

    /**
     * Publishes a {@code PET_DETECTED} overlay event, atomically debounced per
     * camera with the given window.
     */
    @Async
    public void publishPet(String cameraName, long debounceMs) {
        publishDebounced(cameraName, debounceMs, MessagesEnum.PET_DETECTED, lastPetPublish);
    }

    /**
     * Atomic per-camera debounce: the decision to publish and the timestamp update
     * happen inside {@link ConcurrentHashMap#compute}, never as a read-then-write.
     */
    private void publishDebounced(
            String cameraName,
            long debounceMs,
            MessagesEnum template,
            Map<String, Long> lastPublishByCamera
    ) {
        long now = System.currentTimeMillis();
        AtomicBoolean shouldPublish = new AtomicBoolean(false);
        lastPublishByCamera.compute(cameraName, (key, last) -> {
            if (last != null && now - last < debounceMs) {
                return last; // suppressed: within the debounce window
            }
            shouldPublish.set(true);
            return now;
        });

        if (shouldPublish.get()) {
            publish(cameraName, template, List.of());
        }
    }
}
