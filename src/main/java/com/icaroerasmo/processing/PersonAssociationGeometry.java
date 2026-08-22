package com.icaroerasmo.processing;

import com.icaroerasmo.model.FaceRecognition;
import org.bytedeco.opencv.opencv_core.Rect;

import java.util.List;

/**
 * Pure geometry helpers for associating person rectangles with face rectangles.
 * Extracted verbatim from {@code RtspRecognitionRunner}.
 */
public final class PersonAssociationGeometry {

    private PersonAssociationGeometry() {
    }

    public static String findIdentityForPerson(Rect personRect, List<FaceRecognition.DetectedFaces> faces) {
        FaceRecognition.DetectedFaces bestFace = null;
        double bestDistance = Double.MAX_VALUE;

        for (FaceRecognition.DetectedFaces face : faces) {
            if (face.getFaceRect() == null || !containsCenter(personRect, face.getFaceRect())) {
                continue;
            }
            if (face.getDistance() < bestDistance) {
                bestFace = face;
                bestDistance = face.getDistance();
            }
        }

        return bestFace != null ? bestFace.getPersonName() : "Unknown";
    }

    public static Rect findPersonRectForFace(Rect faceRect, List<Rect> detectedPeople) {
        if (faceRect == null) {
            return null;
        }

        Rect bestMatch = null;
        long smallestArea = Long.MAX_VALUE;
        for (Rect personRect : detectedPeople) {
            if (!containsCenter(personRect, faceRect)) {
                continue;
            }

            long area = (long) personRect.width() * personRect.height();
            if (area < smallestArea) {
                bestMatch = personRect;
                smallestArea = area;
            }
        }

        return bestMatch != null ? bestMatch : faceRect;
    }

    public static boolean containsCenter(Rect outerRect, Rect innerRect) {
        int centerX = innerRect.x() + innerRect.width() / 2;
        int centerY = innerRect.y() + innerRect.height() / 2;
        return centerX >= outerRect.x()
            && centerX <= outerRect.x() + outerRect.width()
            && centerY >= outerRect.y()
            && centerY <= outerRect.y() + outerRect.height();
    }
}
