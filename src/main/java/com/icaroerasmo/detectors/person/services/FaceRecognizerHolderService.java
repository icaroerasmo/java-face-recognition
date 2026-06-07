package com.icaroerasmo.detectors.person.services;

import lombok.extern.log4j.Log4j2;
import org.bytedeco.opencv.opencv_face.FaceRecognizer;
import org.springframework.stereotype.Service;

import java.util.concurrent.atomic.AtomicReference;

/**
 * Thread-safe holder for the current FaceRecognizer instance.
 * Allows the recognizer to be updated dynamically when training data changes.
 */
@Log4j2
@Service
public class FaceRecognizerHolderService {

    private final AtomicReference<FaceRecognizer> recognizerRef = new AtomicReference<>();

    /**
     * Get the current FaceRecognizer instance
     * @return the current recognizer, or null if not initialized
     */
    public FaceRecognizer get() {
        return recognizerRef.get();
    }

    /**
     * Update the FaceRecognizer instance atomically
     * @param newRecognizer the new recognizer to use
     * @return the old recognizer that was replaced
     */
    public FaceRecognizer updateRecognizer(FaceRecognizer newRecognizer) {
        FaceRecognizer oldRecognizer = recognizerRef.getAndSet(newRecognizer);
        
        if (oldRecognizer != null) {
            log.info("FaceRecognizer updated - old instance will be closed");
            try {
                // Close the old recognizer to free resources
                oldRecognizer.close();
            } catch (Exception e) {
                log.warn("Error closing old FaceRecognizer", e);
            }
        } else {
            log.info("FaceRecognizer initialized for the first time");
        }
        
        return oldRecognizer;
    }

    /**
     * Check if a recognizer is currently set
     */
    public boolean isInitialized() {
        return recognizerRef.get() != null;
    }
}
