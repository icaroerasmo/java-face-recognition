package com.icaroerasmo.service;

import com.icaroerasmo.model.GifFrame;
import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.apache.commons.collections4.QueueUtils;
import org.apache.commons.collections4.queue.CircularFifoQueue;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

import static org.bytedeco.ffmpeg.global.avutil.AV_LOG_PANIC;
import static org.bytedeco.ffmpeg.global.avutil.av_log_set_level;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;

@Log4j2
@Service
public class RtspFrameExtractorService {

    @SneakyThrows
    public void extract(String rtspUrl, Consumer<Mat> consumer) {
        av_log_set_level(AV_LOG_PANIC);

        FFmpegFrameGrabber grabber = new FFmpegFrameGrabber(rtspUrl);
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();

        try {

            grabber.start();

            while (true) {
                Frame frame = null;
                try {
                    frame = grabber.grab();

                    // End of stream
                    if (frame == null) {
                        break;
                    }

                    if (frame.image != null) {
                        // Convert frame to Mat and clone it so the Mat is independent of the Frame's native memory.
                        Mat nativeMat = (Mat) converter.convert(frame);
                        Mat img = nativeMat.clone();

                        // release the nativeMat copy (backed by Frame) to avoid holding native buffers
                        if (nativeMat != null) {
                            try { nativeMat.release(); } catch (Exception ignore) {}
                        }

                        // Pass cloned Mat to consumer. Consumer is responsible for releasing the Mat when done.
                        consumer.accept(img);
                    }
                } finally {
                    // Ensure frame native resources are released
                    if (frame != null) {
                        try { frame.close(); } catch (Exception ignore) {}
                    }
                }
            }

        } catch (FFmpegFrameGrabber.Exception e) {
            throw new RuntimeException(e);
        } catch (IOException e) {
            throw new RuntimeException(e);
        } finally {
            // Ensure grabber is stopped and released to free native resources
            try {
                if (grabber != null) {
                    try { grabber.stop(); } catch (Exception ignore) {}
                    try { grabber.release(); } catch (Exception ignore) {}
                    try { grabber.close(); } catch (Exception ignore) {}
                }
            } catch (Exception ignored) {
            }

            // converter doesn't have an explicit close, but help GC by nulling reference (no-op here)
            converter = null;

            // Suggest running finalization and GC to help free native memory more quickly
            try {
                System.runFinalization();
                System.gc();
            } catch (Exception ignore) {}
        }
    }
}
