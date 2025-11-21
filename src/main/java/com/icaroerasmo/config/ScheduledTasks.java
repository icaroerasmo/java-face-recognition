package com.icaroerasmo.config;

import com.icaroerasmo.service.DetectionHistoryService;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.springframework.scheduling.annotation.EnableScheduling;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

/**
 * Scheduled tasks for application maintenance
 */
@Log4j2
@Component
@EnableScheduling
@RequiredArgsConstructor
public class ScheduledTasks {

    private final DetectionHistoryService detectionHistoryService;

    /**
     * Cleanup old detection records every 10 minutes
     * This prevents memory leaks from accumulating detection history
     */
    @Scheduled(fixedRate = 10 * 60 * 1000) // 10 minutes
    public void cleanupDetectionHistory() {
        log.debug("Running scheduled cleanup of detection history");
        detectionHistoryService.cleanupOldRecords();
    }
}
