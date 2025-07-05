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

        try {

            grabber.start();

            OpenCVFrameConverter converter = new OpenCVFrameConverter.ToMat();

            while(!grabber.isCloseInputStream()) {
                Frame frame = grabber.grab();
                if (frame != null && frame.image != null) {
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
