package com.icaroerasmo.detectors.movement;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Hermetic tests for the pure per-camera debounce/throttle policy.
 */
class MovementAlertPolicyTest {

    @Test
    void firstEventPublishes() {
        MovementAlertPolicy policy = new MovementAlertPolicy();
        assertTrue(policy.shouldPublish("cam", 1000, 5000));
    }

    @Test
    void eventsWithinWindowAreSuppressed() {
        MovementAlertPolicy policy = new MovementAlertPolicy();
        assertTrue(policy.shouldPublish("cam", 1000, 5000));
        assertFalse(policy.shouldPublish("cam", 2000, 5000));
        assertFalse(policy.shouldPublish("cam", 5999, 5000)); // just inside the window
    }

    @Test
    void eventExactlyAtWindowBoundaryPublishes() {
        MovementAlertPolicy policy = new MovementAlertPolicy();
        assertTrue(policy.shouldPublish("cam", 1000, 5000));
        assertTrue(policy.shouldPublish("cam", 6000, 5000)); // 6000 - 1000 == 5000
    }

    @Test
    void eventAfterWindowPublishes() {
        MovementAlertPolicy policy = new MovementAlertPolicy();
        assertTrue(policy.shouldPublish("cam", 1000, 5000));
        assertTrue(policy.shouldPublish("cam", 8000, 5000));
    }

    @Test
    void camerasAreIndependent() {
        MovementAlertPolicy policy = new MovementAlertPolicy();
        assertTrue(policy.shouldPublish("camA", 1000, 5000));
        assertTrue(policy.shouldPublish("camB", 1001, 5000));
        // Both cameras are inside their own window now.
        assertFalse(policy.shouldPublish("camA", 2000, 5000));
        assertFalse(policy.shouldPublish("camB", 3000, 5000));
        // Each camera reaches its window boundary independently.
        assertTrue(policy.shouldPublish("camA", 6000, 5000)); // 6000 - 1000 == 5000
        assertTrue(policy.shouldPublish("camB", 6001, 5000)); // 6001 - 1001 == 5000
    }

    @Test
    void resetClearsCameraWindow() {
        MovementAlertPolicy policy = new MovementAlertPolicy();
        assertTrue(policy.shouldPublish("cam", 1000, 5000));
        assertFalse(policy.shouldPublish("cam", 2000, 5000));
        policy.reset("cam");
        assertTrue(policy.shouldPublish("cam", 2500, 5000));
    }

    @Test
    void resetAllClearsAllCameras() {
        MovementAlertPolicy policy = new MovementAlertPolicy();
        assertTrue(policy.shouldPublish("camA", 0, 5000));
        assertTrue(policy.shouldPublish("camB", 0, 5000));
        policy.resetAll();
        assertTrue(policy.shouldPublish("camA", 10, 5000));
        assertTrue(policy.shouldPublish("camB", 10, 5000));
    }

    @Test
    void throttleIsIndependentFromOverlayDebounce() {
        MovementAlertPolicy policy = new MovementAlertPolicy();
        long overlayDebounce = 5000;
        long throttle = 30000;

        assertTrue(policy.shouldPublish("cam", 0, overlayDebounce));
        assertTrue(policy.shouldSend("cam", 0, throttle));

        // Overlay re-publishes at 6s and 12s; notification stays throttled.
        assertTrue(policy.shouldPublish("cam", 6000, overlayDebounce));
        assertFalse(policy.shouldSend("cam", 6000, throttle));

        assertTrue(policy.shouldPublish("cam", 12000, overlayDebounce));
        assertFalse(policy.shouldSend("cam", 12000, throttle));

        // Notification throttled until 30s after the first send.
        assertFalse(policy.shouldSend("cam", 29999, throttle));
        assertTrue(policy.shouldSend("cam", 30000, throttle));

        // Both windows still independent afterwards.
        assertTrue(policy.shouldPublish("cam", 30000, overlayDebounce));
        assertFalse(policy.shouldSend("cam", 30001, throttle));
    }

    @Test
    void overlayDebounceDoesNotSuppressNotification() {
        MovementAlertPolicy policy = new MovementAlertPolicy();
        long overlayDebounce = 30000;
        long throttle = 5000;

        assertTrue(policy.shouldPublish("cam", 0, overlayDebounce));
        assertTrue(policy.shouldSend("cam", 0, throttle));

        // Notification can send again at 6s even though the overlay window is still open.
        assertTrue(policy.shouldSend("cam", 6000, throttle));
        assertFalse(policy.shouldPublish("cam", 6000, overlayDebounce));
    }
}
