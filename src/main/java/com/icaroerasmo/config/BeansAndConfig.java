package com.icaroerasmo.config;

import com.icaroerasmo.properties.TelegramProperties;
import com.pengrad.telegrambot.TelegramBot;
import lombok.extern.log4j.Log4j2;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Log4j2
@Configuration
public class BeansAndConfig {

    @Bean
    public TelegramBot telegramBot(TelegramProperties telegramProperties) {
        if (telegramProperties.getBotToken() == null || telegramProperties.getBotToken().isBlank()) {
            log.error("Telegram bot token is not configured. Please set face-recognition.telegram.bot-token");
            System.exit(1);
        }

        if (telegramProperties.getChatId() == null || telegramProperties.getChatId().isBlank()) {
            log.error("Telegram chat ID is not configured. Please set face-recognition.telegram.chat-id");
            System.exit(1);
        }

        try {
            TelegramBot bot = new TelegramBot(telegramProperties.getBotToken());
            log.info("Telegram bot initialized successfully with chat ID: {}", telegramProperties.getChatId());
            return bot;
        } catch (Exception e) {
            log.error("Error initializing Telegram bot", e);
            System.exit(1);
            throw new RuntimeException(e);
        }
    }
}
