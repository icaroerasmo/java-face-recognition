package com.icaroerasmo.service;

import com.icaroerasmo.enums.MessagesEnum;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.springframework.context.MessageSource;
import org.springframework.stereotype.Service;

import java.util.Locale;

@Log4j2
@Service
@RequiredArgsConstructor
public class TranslationService {

    private final MessageSource messageSource;

    public String getMessage(MessagesEnum key, Object... params) {
        try {
            return messageSource.getMessage(key.name(), params, Locale.getDefault());
        } catch (Exception e) {
            log.warn("Translation lookup failed for {}: {}", key.name(), e.getMessage());
            return key.name();
        }
    }
}
