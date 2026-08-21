package com.icaroerasmo.messaging;

import com.icaroerasmo.enums.MessagesEnum;
import com.icaroerasmo.messaging.NotificationMessage.MediaType;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.stereotype.Service;

import java.util.Arrays;
import java.util.List;
import java.util.UUID;

@Log4j2
@Service
@RequiredArgsConstructor
public class NotificationPublisher {

    private static final String SENDER = "face-recognition";
    private static final String EXCHANGE = "telegram.exchange";
    private static final String ROUTING_KEY = "telegram.notifications";

    private final RabbitTemplate rabbitTemplate;

    public void publishText(MessagesEnum template, Object... args) {
        List<String> stringArgs = args == null
                ? List.of()
                : Arrays.stream(args).map(String::valueOf).toList();

        NotificationMessage message = new NotificationMessage(
                UUID.randomUUID().toString(),
                SENDER,
                MediaType.TEXT,
                template.name(),
                stringArgs,
                null,
                null,
                null,
                false);

        log.debug("Publishing TEXT notification: template={}, args={}", template.name(), stringArgs);
        rabbitTemplate.convertAndSend(EXCHANGE, ROUTING_KEY, message);
    }

    public void publishPhoto(String rawHtml, byte[] payload) {
        NotificationMessage message = new NotificationMessage(
                UUID.randomUUID().toString(),
                SENDER,
                MediaType.PHOTO,
                null,
                null,
                rawHtml,
                null,
                payload,
                false);

        log.debug("Publishing PHOTO notification: payload={} bytes", payload != null ? payload.length : 0);
        rabbitTemplate.convertAndSend(EXCHANGE, ROUTING_KEY, message);
    }

    public void publishAnimation(String rawHtml, byte[] payload) {
        NotificationMessage message = new NotificationMessage(
                UUID.randomUUID().toString(),
                SENDER,
                MediaType.ANIMATION,
                null,
                null,
                rawHtml,
                null,
                payload,
                false);

        log.debug("Publishing ANIMATION notification: payload={} bytes", payload != null ? payload.length : 0);
        rabbitTemplate.convertAndSend(EXCHANGE, ROUTING_KEY, message);
    }
}
