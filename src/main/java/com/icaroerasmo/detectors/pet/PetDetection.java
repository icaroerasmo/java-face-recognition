package com.icaroerasmo.detectors.pet;

import org.bytedeco.opencv.opencv_core.Rect;

/**
 * A labeled pet detection ({@code "Dog"} or {@code "Cat"}). The {@link Rect} is
 * owned by the caller and must be deallocated once consumed.
 */
public record PetDetection(String label, Rect rect) {
}
