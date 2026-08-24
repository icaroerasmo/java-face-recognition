package com.icaroerasmo.pipeline;

import com.icaroerasmo.detectors.movement.MovementAlertPolicy;
import com.icaroerasmo.detectors.movement.MovementDetector;
import com.icaroerasmo.detectors.movement.MovementResultStore;
import com.icaroerasmo.properties.CameraProperties;
import com.icaroerasmo.service.RtspFrameExtractorService;
import com.icaroerasmo.service.TelegramPublisherService;
import org.springframework.beans.factory.annotation.Qualifier;

/**
 * Concrete worker for a single RTSP camera. Binds the {@link CameraProperties}
 * to the template lifecycle implemented by {@link CameraStreamWorker}.
 *
 * <p>On every stream (re)connect, the per-camera movement reference state and the
 * movement/pet alert windows are reset so the new stream starts with a fresh
 * baseline.
 */
public class RtspCameraStreamWorker extends CameraStreamWorker {

    private final CameraProperties cameraProperties;
    private final CameraPipeline cameraPipeline;
    private final MovementDetector movementDetector;
    private final MovementResultStore movementResultStore;
    private final MovementAlertPolicy movementAlertPolicy;
    private final MovementAlertPolicy petAlertPolicy;

    public RtspCameraStreamWorker(
            CameraProperties cameraProperties,
            RtspFrameExtractorService rtspFrameExtractorService,
            TelegramPublisherService telegramPublisherService,
            CameraPipeline cameraPipeline,
            MovementDetector movementDetector,
            MovementResultStore movementResultStore,
            @Qualifier("movementAlertPolicy") MovementAlertPolicy movementAlertPolicy,
            @Qualifier("petAlertPolicy") MovementAlertPolicy petAlertPolicy
    ) {
        super(rtspFrameExtractorService, telegramPublisherService);
        this.cameraProperties = cameraProperties;
        this.cameraPipeline = cameraPipeline;
        this.movementDetector = movementDetector;
        this.movementResultStore = movementResultStore;
        this.movementAlertPolicy = movementAlertPolicy;
        this.petAlertPolicy = petAlertPolicy;
    }

    @Override
    protected String cameraName() {
        return cameraProperties.getName() != null ? cameraProperties.getName() : "unknown";
    }

    @Override
    protected String rtspUrl() {
        return cameraProperties.getUrl();
    }

    @Override
    protected CameraProperties.TransportProtocol transportProtocol() {
        return cameraProperties.getProtocol();
    }

    @Override
    protected CameraPipeline cameraPipeline() {
        return cameraPipeline;
    }

    @Override
    protected void onStreamConnect() {
        String camera = cameraName();
        movementDetector.reset(camera);
        movementResultStore.reset(camera);
        movementAlertPolicy.reset(camera);
        petAlertPolicy.reset(camera);
    }
}
