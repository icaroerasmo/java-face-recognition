package com.icaroerasmo.service;

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

    @SneakyThrows
    public void extract(String rtspUrl, Consumer<Mat> consumer) {
        av_log_set_level(AV_LOG_PANIC);

        FFmpegFrameGrabber grabber = new FFmpegFrameGrabber(rtspUrl);
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();

        try {
            // Configure for higher FPS and reduced data loss
            grabber.setOption("rtsp_transport", "tcp");
            grabber.setFrameRate(30);
            grabber.setOption("probesize", "32");
            grabber.setOption("analyzeduration", "0");

            // Buffer settings to prevent frame drops
            grabber.setOption("buffer_size", "2048000");
            grabber.setOption("max_delay", "500000");
            grabber.setOption("reorder_queue_size", "1000");

            // Low latency flags
            grabber.setOption("fflags", "nobuffer+fastseek+flush_packets");
            grabber.setOption("flags", "low_delay");

            // Timeout settings
            grabber.setOption("timeout", "10000000");
            grabber.setOption("stimeout", "5000000");

            // Thread settings for better performance
            grabber.setOption("threads", "auto");

            log.info("Starting RTSP grabber with 30 FPS, TCP transport, and optimized buffering for: {}", rtspUrl);
            grabber.start();

            int nullFrameCount = 0;

            while (true) {
                Frame frame = null;
                try {
                    frame = grabber.grab();

                    if (frame == null) {
                        nullFrameCount++;
                        if (nullFrameCount > 100) {
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

        } catch (FFmpegFrameGrabber.Exception e) {
            throw new RuntimeException(e);
        } finally {
            try {
                if (grabber != null) {
                    try { grabber.stop(); } catch (Exception ignore) {}
                    try { grabber.release(); } catch (Exception ignore) {}
                    try { grabber.close(); } catch (Exception ignore) {}
                }
            } catch (Exception ignored) {
            }

            converter = null;

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

