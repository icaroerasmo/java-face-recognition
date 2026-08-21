package com.icaroerasmo.config;

import com.icaroerasmo.properties.GeneralProperties;
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
