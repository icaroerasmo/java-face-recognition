package com.icaroerasmo.config;

import com.icaroerasmo.properties.GeneralProperties;
import com.icaroerasmo.properties.TelegramProperties;
import com.pengrad.telegrambot.TelegramBot;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.springframework.context.MessageSource;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.support.ReloadableResourceBundleMessageSource;
import org.springframework.util.StringUtils;

import java.util.Locale;

@Log4j2
@Configuration
@RequiredArgsConstructor
public class BeansAndConfig {

    private final GeneralProperties generalProperties;

    @PostConstruct
    public void init() {
        setLocale();
    }

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

    @Bean
    public MessageSource messageSource() {
        ReloadableResourceBundleMessageSource messageSource = new ReloadableResourceBundleMessageSource();
        messageSource.setBasename("classpath:messages");
        messageSource.setDefaultEncoding("UTF-8");
        messageSource.setCacheSeconds(3600);
        return messageSource;
    }

    private void setLocale() {
        if (StringUtils.hasText(generalProperties.getLocale())) {
            Locale.setDefault(Locale.forLanguageTag(generalProperties.getLocale()));
            log.info("Locale set to: {}", generalProperties.getLocale());
        }
    }
}
