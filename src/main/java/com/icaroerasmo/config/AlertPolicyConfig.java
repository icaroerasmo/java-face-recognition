package com.icaroerasmo.config;

import com.icaroerasmo.detectors.movement.MovementAlertPolicy;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * Two distinct {@link MovementAlertPolicy} instances: one for movement alerts and
 * one for pet alerts, so their debounce/throttle windows never interfere.
 */
@Configuration
public class AlertPolicyConfig {

    @Bean
    public MovementAlertPolicy movementAlertPolicy() {
        return new MovementAlertPolicy();
    }

    @Bean
    public MovementAlertPolicy petAlertPolicy() {
        return new MovementAlertPolicy();
    }
}
