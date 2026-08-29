package com.icaroerasmo.detectors.shared;

import com.icaroerasmo.detectors.person.helper.DnnInferenceCoordinatorHelper;
import com.icaroerasmo.detectors.shared.engine.DnnEngine;
import com.icaroerasmo.detectors.shared.engine.DnnEngineFactory;
import com.icaroerasmo.detectors.shared.engine.TensorData;
import com.icaroerasmo.properties.ObjectDetectionProperties;
import com.icaroerasmo.utils.MatUtil;
import com.icaroerasmo.utils.OpenCvResourceHelper;
import lombok.extern.log4j.Log4j2;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.RectVector;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

import static org.bytedeco.opencv.global.opencv_core.BORDER_CONSTANT;
import static org.bytedeco.opencv.global.opencv_core.CV_32F;
import static org.bytedeco.opencv.global.opencv_core.copyMakeBorder;
import static org.bytedeco.opencv.global.opencv_dnn.NMSBoxes;
import static org.bytedeco.opencv.global.opencv_dnn.blobFromImage;
import static org.bytedeco.opencv.global.opencv_imgproc.INTER_LINEAR;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;

/**
 * YOLOv8n (ONNX, COCO) detector. Loads {@code opencv/yolov8n.onnx} exactly ONCE and
 * exposes raw COCO detections. Both {@code PersonDetector} and {@code PetDetector}
 * consume this instance so only a single model copy ever exists in memory.
 *
 * <p>Every DNN forward pass runs inside
 * {@link DnnInferenceCoordinatorHelper#runExclusive} (person, face and pet share the
 * inference coordinator). The returned {@link Rect}s are owned by the caller and must
 * be deallocated when consumed; every Rect created here but not returned (NMS
 * suppressed) is deallocated before this method returns.
 *
 * <p>The inference engine (OpenCV DNN or ONNX Runtime) is selected by
 * {@link DnnEngineFactory} based on the configured acceleration backend.
 */
@Log4j2
@Service
public class YoloDetector {

    private static final String MODEL_FILE = "opencv/yolov8n.onnx";
    private static final int INPUT_SIZE = 640;
    // Output layout: [1, 84, 8400] CV_32F -> 84 = 4 bbox coords + 80 COCO class scores, 8400 anchors.
    private static final int OUTPUT_ROWS = 84;
    private static final int OUTPUT_COLS = 8400;
    private static final int CLASS_COUNT = 80;

    // Low confidence floor for the raw detections; actual per-class filtering happens
    // downstream in PersonDetector/PetDetector.
    private static final float DETECT_MIN_CONFIDENCE = 0.25f;
    private static final float NMS_THRESHOLD = 0.45f;
    private static final float PAD_VALUE = 114.0f;

    // Diagnostic logging: at most one RAW_DET line every 2s, listing only detections
    // above this confidence floor. Purely observational - never changes detection.
    private static final long RAW_LOG_INTERVAL_MS = 2000;
    private static final float RAW_LOG_MIN_CONFIDENCE = 0.10f;

    private volatile long lastRawLogTime = 0L;

    private final DnnEngine engine;
    private final MatUtil matUtil;
    private final DnnInferenceCoordinatorHelper dnnInferenceCoordinatorHelper;

    public YoloDetector(
            MatUtil matUtil,
            ObjectDetectionProperties objectDetectionProperties,
            DnnInferenceCoordinatorHelper dnnInferenceCoordinatorHelper
    ) {
        this.matUtil = matUtil;
        this.dnnInferenceCoordinatorHelper = dnnInferenceCoordinatorHelper;
        try {
            String modelPath = OpenCvResourceHelper.getResourcePath(MODEL_FILE, YoloDetector.class);

            log.info("Loading YOLOv8n model from: {}", modelPath);
            this.engine = DnnEngineFactory.create(
                    modelPath,
                    objectDetectionProperties.getAcceleration().getBackend(),
                    objectDetectionProperties.getAcceleration().getPersonDetectionTarget(),
                    objectDetectionProperties.getAcceleration().isFallbackToCpu(),
                    "object detection"
            );
            log.info("YOLOv8n model loaded successfully [{}]", engine.describe());
        } catch (Exception e) {
            log.error("Failed to load YOLOv8n model: {}", e.getMessage(), e);
            throw new RuntimeException("Failed to initialize YOLOv8n model", e);
        }
    }

    /**
     * Runs the YOLOv8n forward pass and returns raw COCO detections with original
     * image pixel rectangles. The returned {@link Rect}s are owned by the caller;
     * NMS-suppressed rects are deallocated here.
     */
    public List<CocoDetection> detectRaw(Mat image) {
        List<CocoDetection> detections = new ArrayList<>();
        if (image == null || image.empty()) {
            log.warn("Cannot run detection on null or empty image");
            return detections;
        }

        int originalWidth = image.cols();
        int originalHeight = image.rows();
        if (originalWidth <= 0 || originalHeight <= 0) {
            return detections;
        }

        Mat resized = null, canvas = null, blob = null;
        Scalar padColor = null, blobMean = null;
        Size resizedSize = null, blobSize = null;
        RectVector boxesVector = null;
        FloatPointer outputPointer = null;
        List<Rect> candidates = new ArrayList<>();
        List<Float> candidateScores = new ArrayList<>();
        List<Integer> candidateClassIds = new ArrayList<>();

        try {
            // --- Letterbox to 640x640 (resize + pad with gray 114) ---
            double r = Math.min((double) INPUT_SIZE / originalHeight, (double) INPUT_SIZE / originalWidth);
            int newW = (int) Math.round(originalWidth * r);
            int newH = (int) Math.round(originalHeight * r);
            double dw = (INPUT_SIZE - newW) / 2.0;
            double dh = (INPUT_SIZE - newH) / 2.0;
            int top = (int) Math.round(dh - 0.1);
            int bottom = (int) Math.round(dh + 0.1);
            int left = (int) Math.round(dw - 0.1);
            int right = (int) Math.round(dw + 0.1);

            resized = new Mat();
            resizedSize = new Size(newW, newH);
            resize(image, resized, resizedSize, 0, 0, INTER_LINEAR);

            canvas = new Mat();
            padColor = new Scalar(PAD_VALUE, PAD_VALUE, PAD_VALUE, 0);
            copyMakeBorder(resized, canvas, top, bottom, left, right, BORDER_CONSTANT, padColor);

            // --- Blob (1/255, swapRB for RGB->BGR models, no crop) ---
            blobSize = new Size(INPUT_SIZE, INPUT_SIZE);
            blobMean = new Scalar(0, 0, 0, 0);
            blob = blobFromImage(canvas, 1.0 / 255.0, blobSize, blobMean, true, false, CV_32F);
            if (blob == null || blob.empty()) {
                log.warn("Failed to create blob from image");
                return detections;
            }

            // --- Forward ---
            Mat inferenceBlob = blob;
            TensorData output = dnnInferenceCoordinatorHelper.runExclusive("object detection", () -> engine.forward(inferenceBlob));

            // --- Parse [1, 84, 8400] ---
            // The tensor is read flat with a row stride of 8400 (equivalent to reshaping
            // to [84, 8400]); no physical Mat reshape is needed and no extra native
            // wrapper is allocated.
            int rows = output.size(1);
            int cols = output.size(2);
            long total = output.total();
            if (rows != OUTPUT_ROWS || cols != OUTPUT_COLS || total != (long) OUTPUT_ROWS * OUTPUT_COLS) {
                log.warn("Unexpected YOLOv8 output shape: {}x{} (total={})", rows, cols, total);
                return detections;
            }

            float[] d = output.data();

            for (int j = 0; j < OUTPUT_COLS; j++) {
                // bbox in 640-scale pixels (already decoded by the model)
                float cx = d[j];
                float cy = d[OUTPUT_COLS + j];
                float w = d[2 * OUTPUT_COLS + j];
                float h = d[3 * OUTPUT_COLS + j];

                // argmax over the 80 class scores
                int bestClass = -1;
                float bestScore = DETECT_MIN_CONFIDENCE;
                int scoreBase = 4 * OUTPUT_COLS + j;
                for (int c = 0; c < CLASS_COUNT; c++) {
                    float score = d[scoreBase + c * OUTPUT_COLS];
                    if (score > bestScore) {
                        bestScore = score;
                        bestClass = c;
                    }
                }
                if (bestClass < 0) {
                    continue; // below the low confidence floor
                }

                // Map back to original image pixels, removing the letterbox padding.
                float x1 = (cx - w / 2 - left) / (float) r;
                float y1 = (cy - h / 2 - top) / (float) r;
                float x2 = (cx + w / 2 - left) / (float) r;
                float y2 = (cy + h / 2 - top) / (float) r;

                int ix1 = clamp((int) Math.floor(x1), 0, originalWidth);
                int iy1 = clamp((int) Math.floor(y1), 0, originalHeight);
                int ix2 = clamp((int) Math.ceil(x2), 0, originalWidth);
                int iy2 = clamp((int) Math.ceil(y2), 0, originalHeight);
                int boxW = ix2 - ix1;
                int boxH = iy2 - iy1;
                if (boxW <= 0 || boxH <= 0) {
                    continue;
                }

                candidates.add(new Rect(ix1, iy1, boxW, boxH));
                candidateScores.add(bestScore);
                candidateClassIds.add(bestClass);
            }

            if (candidates.isEmpty()) {
                return detections;
            }

            // --- NMS ---
            boxesVector = new RectVector(candidates.size());
            float[] scores = new float[candidates.size()];
            for (int i = 0; i < candidates.size(); i++) {
                boxesVector.put(i, candidates.get(i));
                scores[i] = candidateScores.get(i);
            }
            // Sentinel-initialized so we can count the kept indices (NMSBoxes writes the
            // kept indices first, without reporting the count).
            int[] kept = new int[candidates.size()];
            Arrays.fill(kept, -1);
            NMSBoxes(boxesVector, scores, DETECT_MIN_CONFIDENCE, NMS_THRESHOLD, kept);

            Set<Rect> returnedRects = Collections.newSetFromMap(new IdentityHashMap<>());
            for (int i = 0; i < kept.length; i++) {
                if (kept[i] < 0) {
                    break;
                }
                int idx = kept[i];
                Rect rect = candidates.get(idx);
                detections.add(new CocoDetection(candidateClassIds.get(idx), candidateScores.get(idx), rect));
                returnedRects.add(rect);
            }

            // NMS-suppressed rects are deallocated here; returned rects are owned by the caller.
            for (Rect rect : candidates) {
                if (!returnedRects.contains(rect)) {
                    rect.deallocate();
                }
            }

            logRawDetections(detections);
            return detections;
        } catch (Exception e) {
            log.error("Error during YOLOv8 detection: {}", e.getMessage(), e);
            // No rects may leave this method; release every candidate rect created so far.
            for (Rect rect : candidates) {
                rect.deallocate();
            }
            return new ArrayList<>();
        } finally {
            if (boxesVector != null) {
                boxesVector.deallocate();
            }
            if (padColor != null) {
                padColor.deallocate();
            }
            if (blobMean != null) {
                blobMean.deallocate();
            }
            if (resizedSize != null) {
                resizedSize.deallocate();
            }
            if (blobSize != null) {
                blobSize.deallocate();
            }
            // outputPointer is a proxy view over output.data() and must NOT be released.
            matUtil.releaseResources(resized, canvas, blob);
        }
    }

    private static int clamp(int value, int min, int max) {
        return Math.max(min, Math.min(value, max));
    }

    /**
     * Throttled INFO diagnostic: one line listing raw detections with class name +
     * confidence (e.g. {@code RAW_DET: person=0.82 dog=0.31 pottedplant=0.12 car=0.11}).
     * One entry per class (highest confidence kept), only detections above
     * {@value #RAW_LOG_MIN_CONFIDENCE}. At most one line per {@value #RAW_LOG_INTERVAL_MS} ms.
     */
    private void logRawDetections(List<CocoDetection> detections) {
        long now = System.currentTimeMillis();
        if (now - lastRawLogTime < RAW_LOG_INTERVAL_MS) {
            return;
        }
        lastRawLogTime = now;

        // Preserve first-seen (model output) order, keeping the best confidence per class.
        Map<Integer, Float> bestByClass = new LinkedHashMap<>();
        for (CocoDetection detection : detections) {
            if (detection.confidence() <= RAW_LOG_MIN_CONFIDENCE) {
                continue;
            }
            bestByClass.merge(detection.classId(), detection.confidence(), Math::max);
        }
        if (bestByClass.isEmpty()) {
            return;
        }

        StringBuilder sb = new StringBuilder("RAW_DET:");
        for (Map.Entry<Integer, Float> entry : bestByClass.entrySet()) {
            sb.append(' ')
              .append(DetectionClassFilter.classIdToName(entry.getKey()))
              .append('=')
              .append(String.format(Locale.ROOT, "%.2f", entry.getValue()));
        }
        log.info(sb.toString());
    }
}