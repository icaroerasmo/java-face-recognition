package com.icaroerasmo.messaging;

import com.icaroerasmo.enums.MessagesEnum;
import com.icaroerasmo.messaging.NotificationMessage.MediaType;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Arrays;
import java.util.List;
import java.util.UUID;

@Log4j2
@Service
@RequiredArgsConstructor
public class NotificationPublisher {
    private static final String SENDER = "object-detection";
    private static final String EXCHANGE = "telegram.exchange";
    private static final String ROUTING_KEY = "telegram.notifications";
    private static final DateTimeFormatter DATE_TIME_FORMATTER = DateTimeFormatter.ofPattern("dd/MM/yyyy HH:mm:ss");

    private final RabbitTemplate rabbitTemplate;

    @Async
    public void publishText(MessagesEnum template, Object... args) {
        List<String> stringArgs = args == null
                ? List.of()
                : Arrays.stream(args).map(String::valueOf).toList();

        String sentAt = LocalDateTime.now().format(DATE_TIME_FORMATTER);

        NotificationMessage message = new NotificationMessage(
                UUID.randomUUID().toString(),
                SENDER,
                MediaType.TEXT,
                template.name(),
                stringArgs,
                null,
                null,
                null,
                null,
                false,
                sentAt);

        log.debug("Publishing TEXT notification: template={}, args={}", template.name(), stringArgs);
        rabbitTemplate.convertAndSend(EXCHANGE, ROUTING_KEY, message);
    }

    @Async
    public void publishPhoto(NotificationMessage.CaptionSpec caption, byte[] payload) {
        String sentAt = LocalDateTime.now().format(DATE_TIME_FORMATTER);

        NotificationMessage message = new NotificationMessage(
                UUID.randomUUID().toString(),
                SENDER,
                MediaType.PHOTO,
                null,
                null,
                null,
                caption,
                null,
                payload,
                false,
                sentAt);

        log.debug("Publishing PHOTO notification: payload={} bytes", payload != null ? payload.length : 0);
        rabbitTemplate.convertAndSend(EXCHANGE, ROUTING_KEY, message);
    }

    /**
     * Publishes a PHOTO notification rendered from a message template (used as the
     * caption) plus template arguments. {@code rawHtml} and {@code caption} are null.
     */
    @Async
    public void publishPhotoWithTemplate(MessagesEnum template, Object[] args, byte[] payload) {
        List<String> stringArgs = args == null
                ? List.of()
                : Arrays.stream(args).map(String::valueOf).toList();

        String sentAt = LocalDateTime.now().format(DATE_TIME_FORMATTER);

        NotificationMessage message = new NotificationMessage(
                UUID.randomUUID().toString(),
                SENDER,
                MediaType.PHOTO,
                template.name(),
                stringArgs,
                null,
                null,
                null,
                payload,
                false,
                sentAt);

        log.debug("Publishing PHOTO notification: template={}, args={}, payload={} bytes",
                template.name(), stringArgs, payload != null ? payload.length : 0);
        rabbitTemplate.convertAndSend(EXCHANGE, ROUTING_KEY, message);
    }

    @Async
    public void publishAnimation(NotificationMessage.CaptionSpec caption, byte[] payload) {
        String sentAt = LocalDateTime.now().format(DATE_TIME_FORMATTER);

        NotificationMessage message = new NotificationMessage(
                UUID.randomUUID().toString(),
                SENDER,
                MediaType.ANIMATION,
                null,
                null,
                null,
                caption,
                null,
                payload,
                false,
                sentAt);

        log.debug("Publishing ANIMATION notification: payload={} bytes", payload != null ? payload.length : 0);
        rabbitTemplate.convertAndSend(EXCHANGE, ROUTING_KEY, message);
    }
}
