package com.icaroerasmo.detectors.movement;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Pure per-camera alert policy combining an overlay debounce window and a separate
 * notification throttle window. Used for both movement and pet alerts (one instance per
 * alert kind) so the notification channel is not spammed even when the overlay updates
 * more frequently.
 *
 * <p>All updates are atomic per camera via {@link ConcurrentHashMap#compute} - never
 * a non-atomic read-then-write. A time source is injected as a {@code long now}
 * parameter for deterministic testing.
 *
 * <p>Semantics: the first event publishes; events within {@code windowMs} are
 * suppressed; different cameras are independent; {@link #reset} clears a camera.
 */
public class MovementAlertPolicy {

    // [0] = last overlay publish time, [1] = last notification send time
    private final ConcurrentHashMap<String, long[]> lastActionByCamera = new ConcurrentHashMap<>();

    /**
     * @return {@code true} when the overlay detection event should be published
     *         (debounced with {@code debounceMs} per camera).
     */
    public boolean shouldPublish(String cameraName, long now, long debounceMs) {
        return updateWindow(cameraName, now, debounceMs, 0);
    }

    /**
     * @return {@code true} when a notification should be sent (throttled
     *         with {@code throttleMs} per camera).
     */
    public boolean shouldSend(String cameraName, long now, long throttleMs) {
        return updateWindow(cameraName, now, throttleMs, 1);
    }

    public void reset(String cameraName) {
        lastActionByCamera.remove(cameraName);
    }

    public void resetAll() {
        lastActionByCamera.clear();
    }

    private boolean updateWindow(String cameraName, long now, long windowMs, int index) {
        AtomicBoolean allowed = new AtomicBoolean(false);
        lastActionByCamera.compute(cameraName, (key, state) -> {
            long[] current = state != null ? state : new long[]{Long.MIN_VALUE, Long.MIN_VALUE};
            long last = current[index];
            // Long.MIN_VALUE is the "never published" sentinel; guarded explicitly to
            // avoid overflow in now - last.
            boolean canPublish = last == Long.MIN_VALUE || now - last >= windowMs;
            if (canPublish) {
                long[] updated = current.clone();
                updated[index] = now;
                allowed.set(true);
                return updated;
            }
            return current;
        });
        return allowed.get();
    }
}
