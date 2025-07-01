package com.icaroerasmo.service;

import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.util.function.Consumer;

import static org.bytedeco.ffmpeg.global.avutil.AV_LOG_PANIC;
import static org.bytedeco.ffmpeg.global.avutil.av_log_set_level;

@Service
public class RtspFrameExtractor {
    public void extract(Consumer<Mat> consumer) {
        String rtspUrl = "rtsp://localhost:8554/backyard?video&audio";

        av_log_set_level(AV_LOG_PANIC);

        FFmpegFrameGrabber grabber = new FFmpegFrameGrabber(rtspUrl);
        try {
            grabber.start();

            OpenCVFrameConverter converter = new OpenCVFrameConverter.ToMat();

            while(!grabber.isCloseInputStream()) {
                Frame frame = grabber.grab();
                if (frame != null && frame.image != null) {
                    // Process the frame (e.g., convert to BufferedImage and save)
                    Mat img = (Mat) converter.convert(frame);
                    consumer.accept(img);
                }
                frame.close();
            }

        } catch (FFmpegFrameGrabber.Exception e) {
            throw new RuntimeException(e);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
