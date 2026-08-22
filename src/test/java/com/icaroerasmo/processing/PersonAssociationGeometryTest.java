package com.icaroerasmo.processing;

import com.icaroerasmo.model.FaceRecognition;
import org.bytedeco.opencv.opencv_core.Rect;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static com.icaroerasmo.processing.PersonAssociationGeometry.containsCenter;
import static com.icaroerasmo.processing.PersonAssociationGeometry.findIdentityForPerson;
import static com.icaroerasmo.processing.PersonAssociationGeometry.findPersonRectForFace;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Hermetic geometry tests (tiny in-memory {@link Rect}s only; no DNN/GPU/RabbitMQ).
 */
class PersonAssociationGeometryTest {

    private final List<Rect> ownedRects = new ArrayList<>();

    @AfterEach
    void releaseRects() {
        for (Rect rect : ownedRects) {
            if (rect != null) {
                rect.deallocate();
            }
        }
        ownedRects.clear();
    }

    private Rect rect(int x, int y, int w, int h) {
        Rect rect = new Rect(x, y, w, h);
        ownedRects.add(rect);
        return rect;
    }

    private static FaceRecognition.DetectedFaces face(String name, double distance, Rect faceRect) {
        return new FaceRecognition.DetectedFaces(name, distance, faceRect);
    }

    @Test
    void containsCenterFaceCenterInside() {
        Rect outer = rect(0, 0, 100, 100);
        Rect inner = rect(40, 40, 20, 20); // center (50, 50)
        assertTrue(containsCenter(outer, inner));
    }

    @Test
    void containsCenterFaceCenterOutside() {
        Rect outer = rect(0, 0, 100, 100);
        Rect inner = rect(150, 150, 20, 20);
        assertFalse(containsCenter(outer, inner));
    }

    @Test
    void containsCenterOnBoundaryIsContained() {
        Rect outer = rect(0, 0, 100, 100);
        Rect inner = rect(90, 90, 20, 20); // center (100, 100) == outer bottom-right edge
        assertTrue(containsCenter(outer, inner));
    }

    @Test
    void findIdentityForPersonReturnsMatchingFace() {
        Rect person = rect(0, 0, 200, 200);
        Rect faceRect = rect(50, 50, 40, 40);
        List<FaceRecognition.DetectedFaces> faces = List.of(face("Alice", 0.3, faceRect));
        assertEquals("Alice", findIdentityForPerson(person, faces));
    }

    @Test
    void findIdentityForPersonPrefersLowestDistance() {
        Rect person = rect(0, 0, 200, 200);
        Rect farFace = rect(10, 10, 30, 30);
        Rect nearFace = rect(80, 80, 30, 30);
        List<FaceRecognition.DetectedFaces> faces = List.of(
                face("Bob", 0.9, farFace),
                face("Alice", 0.2, nearFace)
        );
        assertEquals("Alice", findIdentityForPerson(person, faces));
    }

    @Test
    void findIdentityForPersonFaceCenterOutsideReturnsUnknown() {
        Rect person = rect(0, 0, 100, 100);
        Rect outsideFace = rect(200, 200, 20, 20);
        assertEquals("Unknown", findIdentityForPerson(person, List.of(face("Alice", 0.2, outsideFace))));
    }

    @Test
    void findIdentityForPersonNullFaceRectIsIgnored() {
        Rect person = rect(0, 0, 200, 200);
        Rect insideFace = rect(50, 50, 20, 20);
        List<FaceRecognition.DetectedFaces> faces = new ArrayList<>();
        faces.add(face("NullFace", 0.1, null));
        faces.add(face("Alice", 0.3, insideFace));
        assertEquals("Alice", findIdentityForPerson(person, faces));
    }

    @Test
    void findIdentityForPersonEmptyFacesReturnsUnknown() {
        assertEquals("Unknown", findIdentityForPerson(rect(0, 0, 100, 100), List.of()));
    }

    @Test
    void findPersonRectForFaceSmallestContainingRectWins() {
        Rect bigPerson = rect(0, 0, 300, 300);
        Rect smallPerson = rect(20, 20, 100, 100);
        Rect face = rect(30, 30, 40, 40); // center inside both
        assertSame(smallPerson, findPersonRectForFace(face, List.of(bigPerson, smallPerson)));
    }

    @Test
    void findPersonRectForFaceFallsBackToFaceRect() {
        Rect face = rect(0, 0, 40, 40);
        Rect unrelatedPerson = rect(500, 500, 60, 60);
        assertSame(face, findPersonRectForFace(face, List.of(unrelatedPerson)));
    }

    @Test
    void findPersonRectForFaceNullFaceReturnsNull() {
        assertNull(findPersonRectForFace(null, List.of(rect(0, 0, 100, 100))));
    }
}
