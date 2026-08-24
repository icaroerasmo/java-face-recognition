package com.icaroerasmo.detectors.movement;

import org.springframework.stereotype.Component;

import java.util.concurrent.ConcurrentHashMap;

/**
 * Holds the latest per-camera movement-detection verdict so the (cheap) frame
 * differencing can run at a high rate in the producer, decoupled from the slower
 * DNN pipeline in the consumer.
 *
 * <p>The consumer reads {@link #isMovementRecent} instead of running the detector
 * itself, so movement alerts are not delayed by person/face inference.
 */
@Component
public class MovementResultStore {

    // How long a "movement detected" verdict stays fresh for consumers (ms). Must be
    // longer than the consumer frame interval so transient movement is not missed.
    private static final long MOVEMENT_TTL_MS = 1000;

    private final ConcurrentHashMap<String, Long> lastMovementByCamera = new ConcurrentHashMap<>();

    /**
     * Records the movement verdict for a camera. A {@code true} verdict (re)arms the
     * freshness window; a {@code false} verdict leaves the existing state untouched
     * so the flag expires naturally via the TTL.
     */
    public void recordMovement(String cameraName, boolean moved) {
        if (moved) {
            lastMovementByCamera.put(cameraName, System.currentTimeMillis());
        }
    }

    /**
     * @return {@code true} when movement was detected for this camera within the last
     *         {@value #MOVEMENT_TTL_MS} ms.
     */
    public boolean isMovementRecent(String cameraName) {
        Long last = lastMovementByCamera.get(cameraName);
        return last != null && System.currentTimeMillis() - last < MOVEMENT_TTL_MS;
    }

    public void reset(String cameraName) {
        lastMovementByCamera.remove(cameraName);
    }
}
