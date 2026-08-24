package com.icaroerasmo.properties;

import org.junit.jupiter.api.Test;
import org.springframework.boot.context.properties.bind.Bindable;
import org.springframework.boot.context.properties.bind.Binder;
import org.springframework.boot.context.properties.source.ConfigurationPropertySources;
import org.springframework.core.env.MapPropertySource;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Property binding/defaults tests using Spring Boot's {@link Binder} with a plain
 * {@link MapPropertySource} - no application context, no network.
 */
class DetectionPropertiesBindingTest {

    /**
     * Binds the whole {@code object-detection} tree. A real anchor property is
     * provided because Spring's {@link Binder} leaves the result unbound when NO
     * property exists under the prefix; the nested defaults are then asserted.
     */
    private ObjectDetectionProperties bind(Map<String, Object> props) {
        Map<String, Object> anchored = new HashMap<>(props);
        anchored.putIfAbsent("object-detection.enabled", "true");
        Binder binder = new Binder(
                ConfigurationPropertySources.from(new MapPropertySource("test", anchored)));
        return binder.bind("object-detection", Bindable.of(ObjectDetectionProperties.class)).get();
    }

    @Test
    void movementDefaultsApplyWhenNoKeysProvided() {
        MovementDetectionProperties movement = bind(Map.of()).getDetection().getMovement();
        assertTrue(movement.isEnabled());
        assertTrue(movement.isNotifyTelegram());
        assertEquals(25, movement.getDifferenceThreshold());
        assertEquals(0.01, movement.getMinMotionRatio(), 1e-9);
        assertEquals(5000L, movement.getDebounceMs());
        assertEquals(30000L, movement.getTelegramThrottleMs());
        assertEquals(320, movement.getProcessingWidth());
        assertEquals(5, movement.getGaussianKernelSize());
        assertEquals(2, movement.getDilationIterations());
    }

    @Test
    void petDefaultsApplyWhenNoKeysProvided() {
        PetDetectionProperties pet = bind(Map.of()).getDetection().getPet();
        assertTrue(pet.isEnabled());
        assertEquals(0.7, pet.getConfidenceThreshold(), 1e-9);
        assertEquals(0.5, pet.getPlantConfidenceThreshold(), 1e-9);
        assertEquals(0.35, pet.getPlantSuppressionIou(), 1e-9);
        assertEquals(5000L, pet.getDebounceMs());
        assertEquals(30000L, pet.getTelegramThrottleMs());
    }

    @Test
    void existingDetectionDefaultsArePreserved() {
        DetectionProperties detection = bind(Map.of()).getDetection();
        assertEquals(0.8, detection.getPersonConfidenceThreshold(), 1e-9);
        assertEquals(0.5, detection.getCarConfidenceThreshold(), 1e-9);
        assertEquals(0.45, detection.getMaxPersonAreaRatio(), 1e-9);
    }

    @Test
    void kebabCaseYamlKeysBindToMovementProperties() {
        Map<String, Object> props = new HashMap<>();
        props.put("object-detection.detection.movement.enabled", "false");
        props.put("object-detection.detection.movement.notify-telegram", "false");
        props.put("object-detection.detection.movement.difference-threshold", "40");
        props.put("object-detection.detection.movement.min-motion-ratio", "0.02");
        props.put("object-detection.detection.movement.debounce-ms", "8000");
        props.put("object-detection.detection.movement.telegram-throttle-ms", "60000");
        props.put("object-detection.detection.movement.processing-width", "640");
        props.put("object-detection.detection.movement.gaussian-kernel-size", "7");
        props.put("object-detection.detection.movement.dilation-iterations", "3");

        MovementDetectionProperties movement = bind(props).getDetection().getMovement();
        assertFalse(movement.isEnabled());
        assertFalse(movement.isNotifyTelegram());
        assertEquals(40, movement.getDifferenceThreshold());
        assertEquals(0.02, movement.getMinMotionRatio(), 1e-9);
        assertEquals(8000L, movement.getDebounceMs());
        assertEquals(60000L, movement.getTelegramThrottleMs());
        assertEquals(640, movement.getProcessingWidth());
        assertEquals(7, movement.getGaussianKernelSize());
        assertEquals(3, movement.getDilationIterations());
    }

    @Test
    void kebabCaseYamlKeysBindToPetProperties() {
        Map<String, Object> props = new HashMap<>();
        props.put("object-detection.detection.pet.enabled", "false");
        props.put("object-detection.detection.pet.confidence-threshold", "0.65");
        props.put("object-detection.detection.pet.plant-confidence-threshold", "0.7");
        props.put("object-detection.detection.pet.plant-suppression-iou", "0.5");
        props.put("object-detection.detection.pet.debounce-ms", "10000");
        props.put("object-detection.detection.pet.telegram-throttle-ms", "45000");

        PetDetectionProperties pet = bind(props).getDetection().getPet();
        assertFalse(pet.isEnabled());
        assertEquals(0.65, pet.getConfidenceThreshold(), 1e-9);
        assertEquals(0.7, pet.getPlantConfidenceThreshold(), 1e-9);
        assertEquals(0.5, pet.getPlantSuppressionIou(), 1e-9);
        assertEquals(10000L, pet.getDebounceMs());
        assertEquals(45000L, pet.getTelegramThrottleMs());
    }
}
