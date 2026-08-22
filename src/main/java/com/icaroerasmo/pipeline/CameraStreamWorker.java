package com.icaroerasmo.pipeline;

import com.icaroerasmo.enums.MessagesEnum;
import com.icaroerasmo.properties.CameraProperties;
import com.icaroerasmo.service.RtspFrameExtractorService;
import com.icaroerasmo.service.TelegramPublisherService;
import lombok.extern.log4j.Log4j2;
import org.bytedeco.opencv.opencv_core.Mat;

/**
 * Template Method: implements the per-camera reconnect / hibernate / exponential
 * backoff lifecycle (previously {@code processCameraStream} in the runner).
 *
 * <p>Concrete subclasses supply the camera identity and the pipeline used to
 * process each frame. Reconnect counters, the 3-failure -> 5-minute hibernate
 * mechanism, the exponential backoff (2s..30s) and all Telegram lifecycle
 * notifications (CAM_RECONNECTING / CAM_CONNECTED / CAM_HIBERNATING /
 * CAM_HIBERNATE_COMPLETE) are preserved exactly.
 */
@Log4j2
public abstract class CameraStreamWorker implements Runnable {

    protected static final int HIBERNATE_AFTER_FAILURES = 3;
    protected static final long HIBERNATE_DURATION_MS = 5 * 60 * 1000; // 5 minutes

    private final RtspFrameExtractorService rtspFrameExtractorService;
    private final TelegramPublisherService telegramPublisherService;

    protected CameraStreamWorker(
            RtspFrameExtractorService rtspFrameExtractorService,
            TelegramPublisherService telegramPublisherService
    ) {
        this.rtspFrameExtractorService = rtspFrameExtractorService;
        this.telegramPublisherService = telegramPublisherService;
    }

    protected abstract String cameraName();

    protected abstract String rtspUrl();

    protected abstract CameraProperties.TransportProtocol transportProtocol();

    protected abstract CameraPipeline cameraPipeline();

    /**
     * Hook invoked before each stream connection attempt. Used by subclasses to
     * reset per-camera state (e.g. movement reference frames and alert windows)
     * on reconnect.
     */
    protected void onStreamConnect() {
        // default: no-op
    }

    /**
     * Infinite reconnection loop with hibernate mechanism for a single camera.
     */
    @Override
    public final void run() {
        String cameraName = cameraName();
        String rtspUrl = rtspUrl();

        log.info("Starting recognition for camera '{}' with {} transport: {}", cameraName, transportProtocol(), rtspUrl);

        // Infinite reconnection loop with hibernate mechanism
        int reconnectAttempt = 0;
        int consecutiveFailures = 0;
        boolean connectionNotified = false; // Track if we've sent connection success notification

        while (true) {
            try {
                if (reconnectAttempt > 0) {
                    log.info("Reconnection attempt #{} for camera '{}'", reconnectAttempt, cameraName);

                    // Send Telegram notification about reconnection attempt
                    try {
                        telegramPublisherService.sendTranslatedMessage(
                            MessagesEnum.CAM_RECONNECTING, cameraName, reconnectAttempt
                        );
                    } catch (Exception e) {
                        log.warn("Failed to send reconnection notification to Telegram: {}", e.getMessage());
                    }
                }

                // Send initial connection notification before starting extraction
                if (!connectionNotified) {
                    try {
                        telegramPublisherService.sendTranslatedMessage(
                            MessagesEnum.CAM_CONNECTED, cameraName
                        );
                        log.info("Camera '{}': Connection established successfully", cameraName);
                        connectionNotified = true;
                    } catch (Exception e) {
                        log.warn("Failed to send connection notification to Telegram: {}", e.getMessage());
                    }
                }

                onStreamConnect();

                rtspFrameExtractorService.extract(rtspUrl, transportProtocol(), this::processFrame);

                // If extract() returns normally, connection was lost
                log.warn("Stream ended for camera '{}' - Connection may have been lost", cameraName);

                // Reset connection notification flag so we can notify on successful reconnection
                connectionNotified = false;

                reconnectAttempt++;
                consecutiveFailures++;

            } catch (Exception e) {
                connectionNotified = false; // Reset flag on error
                reconnectAttempt++;
                consecutiveFailures++;
                log.error("Error with camera '{}' (attempt #{}): {}", cameraName, reconnectAttempt, e.getMessage());
            }

            // Check if we need to hibernate after 3 consecutive failures
            if (consecutiveFailures >= HIBERNATE_AFTER_FAILURES) {
                log.warn("Camera '{}': {} consecutive failures detected. Entering hibernate mode for {} minutes...",
                    cameraName, HIBERNATE_AFTER_FAILURES, HIBERNATE_DURATION_MS / 60000);

                // Send hibernate notification to Telegram
                try {
                    telegramPublisherService.sendTranslatedMessage(
                        MessagesEnum.CAM_HIBERNATING, cameraName, HIBERNATE_AFTER_FAILURES
                    );
                } catch (Exception e) {
                    log.warn("Failed to send hibernate notification to Telegram: {}", e.getMessage());
                }

                // Hibernate for 5 minutes
                try {
                    Thread.sleep(HIBERNATE_DURATION_MS);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    log.error("Camera '{}' hibernate interrupted", cameraName);
                    return;
                }

                // Send wake-up notification
                try {
                    telegramPublisherService.sendTranslatedMessage(
                        MessagesEnum.CAM_HIBERNATE_COMPLETE, cameraName
                    );
                } catch (Exception e) {
                    log.warn("Failed to send wake-up notification to Telegram: {}", e.getMessage());
                }

                log.info("Camera '{}': Hibernate complete. Resuming connection attempts...", cameraName);

                // Reset consecutive failures counter after hibernate
                consecutiveFailures = 0;

            } else {
                // Normal exponential backoff (2s, 4s, 8s, 16s, max 30s)
                long delayMs = Math.min(30000, 2000 * (long) Math.pow(2, Math.min(reconnectAttempt - 1, 4)));
                log.info("Waiting {}ms before reconnecting camera '{}'...", delayMs, cameraName);

                try {
                    Thread.sleep(delayMs);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    log.error("Camera '{}' reconnection interrupted", cameraName);
                    return;
                }
            }
        }
    }

    /**
     * Per-frame entry point: wraps pipeline processing in a try-with-resources
     * {@link FrameContext} so the context remains the single owner of the
     * transferred Rects regardless of success or failure.
     */
    protected void processFrame(Mat img) {
        if (img == null) {
            return;
        }

        try (FrameContext frameContext = new FrameContext(cameraName(), img)) {
            cameraPipeline().process(frameContext);
        } catch (Exception e) {
            log.error("Error processing frame from camera '{}': {}", cameraName(), e.getMessage(), e);
        }
    }
}
