package com.icaroerasmo.service;

import com.icaroerasmo.properties.CameraProperties;
import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.springframework.stereotype.Service;

import java.util.function.Consumer;

import static org.bytedeco.ffmpeg.global.avutil.AV_LOG_PANIC;
import static org.bytedeco.ffmpeg.global.avutil.av_log_set_level;

@Log4j2
@Service
public class RtspFrameExtractorService {

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
                // Set format FIRST before any other options
                grabber.setFormat("rtsp");

                grabber.setFrameRate(30);

                // Configure transport protocol (TCP or UDP) - must be set early
                String transport = transportProtocol != null
                    ? transportProtocol.name().toLowerCase()
                    : "tcp";
                grabber.setOption("rtsp_transport", transport);

                // Connection and timeout options
                grabber.setOption("timeout", "20000000");         // 20 seconds in microseconds
                grabber.setOption("stimeout", "20000000");        // Socket timeout
                grabber.setOption("recv_timeout", "20000000");    // Receive timeout

                // Probing options - must come before frame rate
                grabber.setOption("probesize", "32");
                grabber.setOption("analyzeduration", "0");

                // Frame rate and buffering
                grabber.setFrameRate(30);
                grabber.setOption("buffer_size", "2048000");
                grabber.setOption("max_delay", "500000");
                grabber.setOption("reorder_queue_size", "1000");

                // Low latency and performance flags
                grabber.setOption("fflags", "nobuffer+fastseek+flush_packets");
                grabber.setOption("flags", "low_delay");

                // Thread settings
                grabber.setOption("threads", "auto");

                // Additional reliability options
                grabber.setOption("allowed_media_types", "video");
                grabber.setOption("rtsp_flags", "prefer_tcp");

                if (retryCount == 0) {
                    log.info("Attempting to start RTSP grabber with {} transport for: {}", transportProtocol, rtspUrl);
                } else {
                    log.info("Retry attempt {} of {} for RTSP stream: {}", retryCount, maxRetries - 1, rtspUrl);
                }

                grabber.start();
                log.info("Successfully started RTSP grabber with 30 FPS and optimized buffering for: {}", rtspUrl);

                // If we got here, connection succeeded - process frames
                processFrames(grabber, converter, rtspUrl, consumer);

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
     * Process frames from an already-connected grabber
     */
    private void processFrames(FFmpegFrameGrabber grabber, OpenCVFrameConverter.ToMat converter,
                              String rtspUrl, Consumer<Mat> consumer) throws Exception {
        int nullFrameCount = 0;
        int maxNullFrames = 150;

        try {
            while (true) {
                Frame frame = null;
                try {
                    frame = grabber.grab();

                    if (frame == null) {
                        nullFrameCount++;
                        if (nullFrameCount > maxNullFrames) {
                            log.warn("Exceeded maximum null frames ({}) for stream: {}", maxNullFrames, rtspUrl);
                            break;
                        }
                        Thread.sleep(10);
                        continue;
                    }

                    nullFrameCount = 0;

                    if (frame.image != null) {
                        if (frame.imageWidth <= 0 || frame.imageHeight <= 0) {
                            continue;
                        }

                        Mat nativeMat = converter.convert(frame);
                        if (nativeMat == null || nativeMat.empty()) {
                            continue;
                        }

                        Mat img = nativeMat.clone();

                        try { nativeMat.release(); } catch (Exception ignore) {}

                        if (img.empty() || img.cols() <= 0 || img.rows() <= 0) {
                            try { img.release(); } catch (Exception ignore) {}
                            continue;
                        }

                        if (isGreyFrame(img)) {
                            try { img.release(); } catch (Exception ignore) {}
                            continue;
                        }

                        consumer.accept(img);
                    }
                } finally {
                    if (frame != null) {
                        try { frame.close(); } catch (Exception ignore) {}
                    }
                }
            }
        } finally {

            try {
                System.gc();
            } catch (Exception ignore) {}
        }
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
