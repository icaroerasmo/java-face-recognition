package com.icaroerasmo.service;

import com.icaroerasmo.properties.CameraProperties;
import com.icaroerasmo.properties.StreamsProperties;
import lombok.RequiredArgsConstructor;
import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.springframework.stereotype.Service;

import java.nio.ByteBuffer;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;

import static com.icaroerasmo.utils.Constants.FPS;
import static org.bytedeco.ffmpeg.global.avutil.AV_LOG_PANIC;
import static org.bytedeco.ffmpeg.global.avutil.av_log_set_level;
import static org.bytedeco.opencv.global.opencv_core.CV_8UC1;
import static org.bytedeco.opencv.global.opencv_core.CV_8UC3;
import static org.bytedeco.opencv.global.opencv_core.CV_8UC4;

@Log4j2
@Service
@RequiredArgsConstructor
public class RtspFrameExtractorService {

    private final StreamsProperties streamsProperties;

    // Poison pill to signal consumer thread to stop
    private static final FrameData POISON_PILL = new FrameData(null, 0, 0, 0);

    /**
     * Frame data stored as byte array to prevent native memory leaks
     * Each camera has its own queue of FrameData objects
     */
    private static class FrameData {
        final byte[] data;
        final int width;
        final int height;
        final int channels;

        FrameData(byte[] data, int width, int height, int channels) {
            this.data = data;
            this.width = width;
            this.height = height;
            this.channels = channels;
        }

        boolean isPoisonPill() {
            return data == null;
        }
    }

    /**
     * Extract frames from RTSP stream with configurable transport protocol
     * @param rtspUrl The RTSP URL
     * @param transportProtocol "tcp" or "udp" (defaults to "tcp" if null)
     * @param consumer Callback to process each frame
     */
    @SneakyThrows
    public void extract(String rtspUrl, CameraProperties.TransportProtocol transportProtocol, Consumer<Mat> consumer) {
        av_log_set_level(AV_LOG_PANIC);

        int maxRetries = 3;
        int retryCount = 0;
        int initialDelayMs = 2000; // 2 seconds

        while (retryCount < maxRetries) {
            FFmpegFrameGrabber grabber = new FFmpegFrameGrabber(rtspUrl);
            OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();

            try {
                // Configure all grabber settings
                configureGrabber(grabber, transportProtocol);

                if (retryCount == 0) {
                    log.info("Attempting to start RTSP grabber with {} transport for: {}", transportProtocol, rtspUrl);
                } else {
                    log.info("Retry attempt {} of {} for RTSP stream: {}", retryCount, maxRetries - 1, rtspUrl);
                }

                grabber.start();

                // Process frames using producer-consumer pattern with byte array storage
                processFramesWithQueue(grabber, converter, rtspUrl, consumer);

                // Normal exit
                return;

            } catch (FFmpegFrameGrabber.Exception e) {
                retryCount++;
                String errorMsg = e.getMessage() != null ? e.getMessage() : "Unknown error";

                if (retryCount < maxRetries) {
                    long delayMs = initialDelayMs * (long) Math.pow(2, retryCount - 1); // Exponential backoff
                    log.warn("RTSP connection failed (attempt {}/{}): {}. Retrying in {}ms...",
                        retryCount, maxRetries, errorMsg, delayMs);

                    try {
                        Thread.sleep(delayMs);
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                        throw new RuntimeException("Interrupted while retrying RTSP connection", ie);
                    }
                } else {
                    log.error("RTSP connection failed after {} attempts. Last error: {}", maxRetries, errorMsg);
                    log.error("URL: {}", rtspUrl);
                    log.error("Possible causes:");
                    log.error("  1. Camera is unreachable from container (network/firewall issue)");
                    log.error("  2. Invalid RTSP URL or credentials");
                    log.error("  3. Camera doesn't support RTSP protocol");
                    log.error("  4. Camera requires specific transport protocol (TCP/UDP)");
                    throw new RuntimeException("Failed to connect to RTSP stream after " + maxRetries + " attempts: " + errorMsg, e);
                }
            } catch (Exception e) {
                retryCount++;
                if (retryCount < maxRetries) {
                    long delayMs = initialDelayMs * (long) Math.pow(2, retryCount - 1);
                    log.warn("Unexpected error (attempt {}/{}): {}. Retrying in {}ms...",
                        retryCount, maxRetries, e.getMessage(), delayMs);
                    try {
                        Thread.sleep(delayMs);
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                        throw new RuntimeException("Interrupted while retrying RTSP connection", ie);
                    }
                } else {
                    log.error("Unexpected error after {} attempts: {}", maxRetries, e.getMessage(), e);
                    throw e;
                }
            } finally {
                try {
                    if (grabber != null) {
                        try { grabber.stop(); } catch (Exception ignore) {}
                        try { grabber.release(); } catch (Exception ignore) {}
                        try { grabber.close(); } catch (Exception ignore) {}
                    }
                } catch (Exception ignored) {}
            }
        }
    }

    /**
     * Configure FFmpegFrameGrabber with all necessary settings
     * @param grabber The FFmpegFrameGrabber instance to configure
     * @param transportProtocol TCP or UDP transport protocol
     */
    private void configureGrabber(FFmpegFrameGrabber grabber, CameraProperties.TransportProtocol transportProtocol) {
        // Set format FIRST before any other options
        grabber.setFormat("rtsp");

        grabber.setFrameRate(FPS);

        // Configure transport protocol (TCP or UDP) - must be set early
        String transport = transportProtocol != null
            ? transportProtocol.name().toLowerCase()
            : "tcp";
        boolean isUdp = "udp".equals(transport);

        grabber.setOption("rtsp_transport", transport);

        // Connection and timeout options
        grabber.setOption("timeout", "20000000");         // 20 seconds in microseconds
        grabber.setOption("stimeout", "20000000");        // Socket timeout
        grabber.setOption("recv_timeout", "20000000");    // Receive timeout

        // Probing options - CRITICAL: Higher values for proper stream analysis
        // 4K stream needs more probing to detect codec parameters correctly
        grabber.setOption("probesize", "50000000");       // 50MB probe size for 4K
        grabber.setOption("analyzeduration", "5000000");  // 5 seconds analysis

        // Frame rate
        grabber.setFrameRate(FPS);

        // Buffer size calculation for 4K support
        // 4K frame (3840x2160) at 30fps with H.264:
        // - Uncompressed: ~25MB per frame
        // - Compressed (H.264): ~2-5MB per frame
        // - Buffer needs to hold multiple frames for smooth playback
        // - UDP needs larger buffer due to packet reordering
        int bufferSize = 50 * 1024 * 1024;
        grabber.setOption("buffer_size", String.valueOf(bufferSize));

        if (isUdp) {
            // UDP-specific settings to handle packet loss and reordering
            grabber.setOption("max_delay", "10000000");        // 10 seconds max delay for UDP
            grabber.setOption("reorder_queue_size", "5000");   // Larger reorder queue for UDP
            grabber.setOption("fifo_size", "500000");          // Large FIFO for UDP buffering

            // UDP packet handling
            grabber.setOption("overrun_nonfatal", "1");        // Don't fail on buffer overrun
            grabber.setOption("fflags", "+genpts+igndts+discardcorrupt"); // Generate PTS, ignore DTS, discard corrupt

            // Error resilience for UDP
            grabber.setOption("err_detect", "ignore_err");     // Ignore decoding errors
            grabber.setOption("skip_frame", "noref");          // Skip non-reference frames if needed

            log.info("Configured UDP-specific settings: buffer={}MB, reorder_queue=5000", bufferSize / (1024 * 1024));
        } else {
            // TCP-specific settings for low latency
            grabber.setOption("max_delay", "500000");          // 0.5 seconds for TCP
            grabber.setOption("reorder_queue_size", "1000");   // Smaller queue for TCP
            grabber.setOption("fflags", "nobuffer+fastseek+flush_packets"); // Low latency flags
            grabber.setOption("rtsp_flags", "prefer_tcp");     // Prefer TCP

            log.info("Configured TCP-specific settings: buffer={}MB, low latency mode", bufferSize / (1024 * 1024));
        }

        // Common flags
        grabber.setOption("flags", "low_delay");
        grabber.setOption("flags2", "+export_mvs");            // Export motion vectors for validation

        // Thread settings
        grabber.setOption("threads", "auto");

        // Additional reliability options
        grabber.setOption("allowed_media_types", "video");
    }

    /**
     * Process frames using producer-consumer pattern with byte array storage
     * Producer thread: Grabs frames and stores as byte arrays in queue
     * Consumer thread: Takes byte arrays, reconstructs Mats, and processes them
     */
    private void processFramesWithQueue(FFmpegFrameGrabber grabber, OpenCVFrameConverter.ToMat converter,
                                       String rtspUrl, Consumer<Mat> consumer) throws Exception {
        // Create bounded queue - one per camera stream
        final int queueCapacity = Math.max(1, streamsProperties.getFrameQueueCapacity());
        final int processingFps = Math.max(1, streamsProperties.getProcessingFps());
        final long minFrameIntervalNs = TimeUnit.SECONDS.toNanos(1) / processingFps;
        BlockingQueue<FrameData> frameQueue = new ArrayBlockingQueue<>(queueCapacity);
        AtomicBoolean shouldStop = new AtomicBoolean(false);
        AtomicBoolean producerError = new AtomicBoolean(false);

        // PRODUCER THREAD: Grabs frames and converts to byte arrays
        Thread producer = new Thread(() -> {
            int nullFrameCount = 0;
            final int maxNullFrames = Math.max(1, streamsProperties.getMaxConsecutiveNullFrames());
            long lastQueuedAtNs = 0;

            try {
                log.info("Frame producer started for: {} (processingFps={}, queueCapacity={})",
                        rtspUrl, processingFps, queueCapacity);

                while (!shouldStop.get() && !Thread.currentThread().isInterrupted()) {
                    Frame frame = null;
                    Mat mat = null;

                    try {
                        frame = grabber.grabImage();

                        if (frame == null) {
                            nullFrameCount++;
                            if (nullFrameCount > maxNullFrames) {
                                log.warn("Too many null frames ({}), stopping stream: {}", maxNullFrames, rtspUrl);
                                producerError.set(true);
                                shouldStop.set(true);
                                break;
                            }
                            Thread.sleep(10);
                            continue;
                        }

                        nullFrameCount = 0;

                        if (frame.image == null || frame.imageWidth <= 0 || frame.imageHeight <= 0) {
                            continue;
                        }

                        // Convert Frame to Mat
                        mat = converter.convert(frame);
                        if (mat == null || mat.empty()) {
                            continue;
                        }

                        // Validate Mat
                        if (mat.cols() <= 0 || mat.rows() <= 0 || mat.data() == null || mat.data().isNull()) {
                            try { mat.release(); } catch (Exception ignore) {}
                            continue;
                        }

                        // Skip grey/blank frames
                        if (isGreyFrame(mat)) {
                            try { mat.release(); } catch (Exception ignore) {}
                            continue;
                        }

                        long nowNs = System.nanoTime();
                        if (nowNs - lastQueuedAtNs < minFrameIntervalNs) {
                            try { mat.release(); } catch (Exception ignore) {}
                            continue;
                        }
                        lastQueuedAtNs = nowNs;

                        // Convert Mat to byte array - THIS PREVENTS MEMORY LEAKS
                        int width = mat.cols();
                        int height = mat.rows();
                        int channels = mat.channels();
                        int totalBytes = width * height * channels;

                        byte[] frameBytes = new byte[totalBytes];

                        // Get buffer and ensure it's positioned correctly
                        ByteBuffer buffer = mat.data().capacity(totalBytes).asByteBuffer();
                        buffer.position(0);
                        buffer.limit(totalBytes);
                        buffer.get(frameBytes, 0, totalBytes);

                        // Release Mat immediately after copying to byte array
                        try { mat.release(); } catch (Exception ignore) {}
                        mat = null;

                        // Store byte array in queue
                        FrameData frameData = new FrameData(frameBytes, width, height, channels);
                        boolean added = frameQueue.offer(frameData);

                        if (!added) {
                            FrameData droppedFrame = frameQueue.poll();
                            if (droppedFrame != null && !droppedFrame.isPoisonPill()) {
                                log.debug("Frame queue full, dropping stale frame for: {}", rtspUrl);
                            }
                            frameQueue.offer(frameData);
                        }

                    } catch (Exception e) {
                        log.error("Error in producer for {}: {}", rtspUrl, e.getMessage(), e);
                        if (mat != null) {
                            try { mat.release(); } catch (Exception ignore) {}
                        }
                    } finally {
                        if (frame != null) {
                            try { frame.close(); } catch (Exception ignore) {}
                        }
                    }
                }
            } catch (Exception e) {
                log.error("Producer thread error for {}: {}", rtspUrl, e.getMessage(), e);
                producerError.set(true);
            } finally {
                try {
                    frameQueue.offer(POISON_PILL, 1, TimeUnit.SECONDS);
                } catch (Exception ignore) {}
                log.info("Frame producer stopped for: {}", rtspUrl);
            }
        }, "FrameProducer-" + rtspUrl.hashCode());

        // CONSUMER THREAD: Takes byte arrays and reconstructs Mats
        Thread consumerThread = new Thread(() -> {
            try {
                log.info("Frame consumer started for: {}", rtspUrl);

                while (!shouldStop.get() && !Thread.currentThread().isInterrupted()) {
                    FrameData frameData = frameQueue.poll(1, TimeUnit.SECONDS);

                    if (frameData == null) {
                        continue;
                    }

                    if (frameData.isPoisonPill()) {
                        log.info("Received stop signal for: {}", rtspUrl);
                        break;
                    }

                    Mat mat = null;
                    try {
                        int matType = resolveMatType(frameData.channels);
                        if (matType < 0) {
                            log.warn("Unsupported channel count {} for frame from {}", frameData.channels, rtspUrl);
                            continue;
                        }

                        // Reconstruct Mat from byte array
                        mat = new Mat(frameData.height, frameData.width, matType);

                        // Ensure the Mat has the correct size
                        int expectedSize = frameData.width * frameData.height * frameData.channels;
                        if (frameData.data.length != expectedSize) {
                            log.warn("Frame data size mismatch: expected {}, got {}", expectedSize, frameData.data.length);
                            continue;
                        }

                        // Copy byte array to Mat
                        ByteBuffer buffer = mat.data().capacity(expectedSize).asByteBuffer();
                        buffer.position(0);
                        buffer.put(frameData.data);

                        // Pass to callback for processing
                        consumer.accept(mat);

                    } catch (Exception e) {
                        log.error("Error in consumer for {}: {}", rtspUrl, e.getMessage(), e);
                    } finally {
                        // Consumer thread releases the Mat after callback completes
                        if (mat != null) {
                            try { mat.release(); } catch (Exception ignore) {}
                        }
                    }
                }
            } catch (Exception e) {
                log.error("Consumer thread error for {}: {}", rtspUrl, e.getMessage(), e);
            } finally {
                frameQueue.clear();
                log.info("Frame consumer stopped for: {}", rtspUrl);
            }
        }, "FrameConsumer-" + rtspUrl.hashCode());

        // Start both threads
        producer.start();
        consumerThread.start();

        try {
            // Wait for both threads to complete
            producer.join();
            consumerThread.join();

            if (producerError.get()) {
                throw new RuntimeException("Stream connection lost");
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            shouldStop.set(true);
            producer.interrupt();
            consumerThread.interrupt();
            throw new RuntimeException("Frame processing interrupted", e);
        } finally {
            // Close converter to release native resources
            try {
                converter.close();
            } catch (Exception e) {
                log.warn("Error closing converter: {}", e.getMessage());
            }
        }
    }

    private int resolveMatType(int channels) {
        return switch (channels) {
            case 1 -> CV_8UC1;
            case 3 -> CV_8UC3;
            case 4 -> CV_8UC4;
            default -> -1;
        };
    }


    private boolean isGreyFrame(Mat img) {
        if (img == null || img.empty()) {
            return true;
        }

        try {
            if (img.cols() < 50 || img.rows() < 50) {
                return true;
            }

            Mat mean = new Mat();
            Mat stddev = new Mat();
            org.bytedeco.opencv.global.opencv_core.meanStdDev(img, mean, stddev);

            if (stddev.data() == null || stddev.empty() || stddev.total() == 0) {
                mean.release();
                stddev.release();
                return false;
            }

            int channels = (int) stddev.total();
            double[] stddevValues = new double[channels];

            try {
                stddev.data().asBuffer().asDoubleBuffer().get(stddevValues);
            } catch (Exception e) {
                mean.release();
                stddev.release();
                return false;
            }

            double avgStdDev = 0;
            for (double val : stddevValues) {
                avgStdDev += val;
            }
            avgStdDev /= stddevValues.length;

            mean.release();
            stddev.release();

            return avgStdDev < 15.0;

        } catch (Exception e) {
            return false;
        }
    }
}
