package com.icaroerasmo.config;

import lombok.extern.log4j.Log4j2;
import org.springframework.amqp.core.Binding;
import org.springframework.amqp.core.BindingBuilder;
import org.springframework.amqp.core.DirectExchange;
import org.springframework.amqp.core.Queue;
import org.springframework.amqp.core.QueueBuilder;
import org.springframework.amqp.rabbit.connection.ConnectionFactory;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.amqp.support.converter.Jackson2JsonMessageConverter;
import org.springframework.amqp.support.converter.MessageConverter;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Log4j2
@Configuration
public class RabbitConfig {

    public static final String TELEGRAM_EXCHANGE = "telegram.exchange";
    public static final String TELEGRAM_QUEUE = "telegram.notifications";
    public static final String TELEGRAM_ROUTING_KEY = "telegram.notifications";
    public static final String TELEGRAM_DLX = "telegram.dlx";
    public static final String TELEGRAM_DLQ_ROUTING_KEY = "telegram.notifications.dlq";

    public static final String DETECTION_EXCHANGE = "detection.exchange";
    public static final String DETECTION_QUEUE = "detection.events";
    public static final String DETECTION_ROUTING_KEY = "detection.events";

    @Bean
    public DirectExchange telegramExchange() {
        return new DirectExchange(TELEGRAM_EXCHANGE, true, false);
    }

    @Bean
    public Queue telegramQueue() {
        return QueueBuilder.durable(TELEGRAM_QUEUE)
                .withArgument("x-dead-letter-exchange", TELEGRAM_DLX)
                .withArgument("x-dead-letter-routing-key", TELEGRAM_DLQ_ROUTING_KEY)
                .build();
    }

    @Bean
    public Binding telegramBinding(DirectExchange telegramExchange, Queue telegramQueue) {
        return BindingBuilder.bind(telegramQueue).to(telegramExchange).with(TELEGRAM_ROUTING_KEY);
    }

    @Bean
    public DirectExchange detectionExchange() {
        return new DirectExchange(DETECTION_EXCHANGE, true, false);
    }

    @Bean
    public Queue detectionQueue() {
        return QueueBuilder.durable(DETECTION_QUEUE).build();
    }

    @Bean
    public Binding detectionBinding(DirectExchange detectionExchange, Queue detectionQueue) {
        return BindingBuilder.bind(detectionQueue).to(detectionExchange).with(DETECTION_ROUTING_KEY);
    }

    @Bean
    public MessageConverter messageConverter() {
        return new Jackson2JsonMessageConverter(
                "com.icaroerasmo",
                "java.util",
                "java.lang",
                "org.springframework.amqp");
    }

    @Bean
    public RabbitTemplate rabbitTemplate(ConnectionFactory connectionFactory, MessageConverter messageConverter) {
        RabbitTemplate rabbitTemplate = new RabbitTemplate(connectionFactory);
        rabbitTemplate.setMessageConverter(messageConverter);
        rabbitTemplate.setMandatory(true);
        rabbitTemplate.setConfirmCallback((correlationData, ack, cause) -> {
            if (ack) {
                log.debug("RabbitMQ message confirmed: {}", correlationData != null ? correlationData.getId() : "null");
            } else {
                log.error("RabbitMQ message NOT confirmed: {}, cause: {}",
                        correlationData != null ? correlationData.getId() : "null", cause);
            }
        });
        rabbitTemplate.setReturnsCallback(returned ->
                log.error("RabbitMQ message returned: exchange={}, routingKey={}, replyCode={}, replyText={}",
                        returned.getExchange(), returned.getRoutingKey(),
                        returned.getReplyCode(), returned.getReplyText()));
        return rabbitTemplate;
    }
}
