package com.icaroerasmo.config;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.context.config.ConfigDataEnvironmentPostProcessor;
import org.springframework.boot.context.properties.bind.Bindable;
import org.springframework.boot.context.properties.bind.Binder;
import org.springframework.boot.env.EnvironmentPostProcessor;
import org.springframework.core.Ordered;
import org.springframework.core.env.ConfigurableEnvironment;
import org.springframework.core.env.MapPropertySource;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Maps the obfuscated {@code rabbitMq.*} config prefix to Spring Boot's
 * {@code spring.rabbitmq.*} so the tech stack is not exposed in config files.
 */
public class RabbitMqPropertyMapper implements EnvironmentPostProcessor, Ordered {

    @Override
    public int getOrder() {
        return ConfigDataEnvironmentPostProcessor.ORDER + 100;
    }

    @Override
    public void postProcessEnvironment(ConfigurableEnvironment environment, SpringApplication application) {
        Map<String, Object> props = Binder.get(environment)
                .bind("rabbit-mq", Bindable.mapOf(String.class, Object.class))
                .orElse(Collections.emptyMap());
        if (props.isEmpty()) {
            return;
        }
        Map<String, Object> mapped = new HashMap<>();
        props.forEach((key, value) -> mapped.put("spring.rabbitmq." + key, value));
        environment.getPropertySources().addFirst(new MapPropertySource("rabbitMq-mapped", mapped));
    }
}