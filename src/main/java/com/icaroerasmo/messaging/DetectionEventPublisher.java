package com.icaroerasmo.messaging;

import com.icaroerasmo.enums.MessagesEnum;
import lombok.extern.log4j.Log4j2;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.UUID;

@Log4j2
@Service
public class DetectionEventPublisher {

    private static final String EXCHANGE = "detection.exchange";
    private static final String ROUTING_KEY = "detection.events";

    @Autowired
    private RabbitTemplate rabbitTemplate;

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
}
