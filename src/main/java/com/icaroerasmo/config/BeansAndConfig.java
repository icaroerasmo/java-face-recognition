package com.icaroerasmo.config;

import com.icaroerasmo.properties.GeneralProperties;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.springframework.context.annotation.Configuration;
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

    private void setLocale() {
        if (StringUtils.hasText(generalProperties.getLocale())) {
            Locale.setDefault(Locale.forLanguageTag(generalProperties.getLocale()));
            log.info("Locale set to: {}", generalProperties.getLocale());
        }
    }
}
