package com.icaroerasmo.service;

import com.icaroerasmo.properties.FaceRecognitionProperties;
import com.icaroerasmo.utils.MatUtil;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * Service to create GIF animations from a sequence of frames
 */
@Log4j2
@Service
@RequiredArgsConstructor
public class GifCreationService {

    private final MatUtil matUtil;
    private final FaceRecognitionProperties faceRecognitionProperties;

    // GIF parameters
    private static final int GIF_WIDTH = 640; // Resize frames to this width for smaller file size
    private static final int MIN_FRAMES = 10; // Minimum frames required to create a meaningful GIF

    /**
     * Create a GIF from a list of frame images
     *
     * @param frameImages List of image byte arrays (JPEG format)
     * @return GIF byte array, or null if creation failed
     */
    public byte[] createGif(List<byte[]> frameImages) {
        if (frameImages == null || frameImages.isEmpty()) {
            log.warn("Cannot create GIF: frame list is null or empty");
            return null;
        }

        if (frameImages.size() < MIN_FRAMES) {
            log.info("Not enough frames to create GIF: {} frames (minimum required: {})",
                frameImages.size(), MIN_FRAMES);
            return null;
        }

        log.info("Creating GIF from {} frames", frameImages.size());

        int maxFrames = getMaxGifFrames();

        // Limit number of frames to keep GIF size manageable
        List<byte[]> framesToUse = frameImages;
        if (frameImages.size() > maxFrames) {
            // Sample frames evenly to get the configured frame budget.
            int step = Math.max(1, frameImages.size() / maxFrames);
            framesToUse = new java.util.ArrayList<>();
            for (int i = 0; i < frameImages.size(); i += step) {
                framesToUse.add(frameImages.get(i));
                if (framesToUse.size() >= maxFrames) break;
            }
            log.info("Sampled {} frames from {} total frames (max {})", framesToUse.size(), frameImages.size(), maxFrames);
        }

        try {
            // Since we can't easily create a proper GIF with OpenCV alone,
            // let's create a video format that Telegram supports (MP4)
            return createVideoMp4(framesToUse);

        } catch (Exception e) {
            log.error("Failed to create GIF: {}", e.getMessage(), e);
            return null;
        }
    }

    /**
     * Create an MP4 video from frames (more reliable than GIF for our use case)
     * Telegram supports MP4 animations
     */
    private byte[] createVideoMp4(List<byte[]> frameImages) {
        org.bytedeco.javacv.FFmpegFrameRecorder recorder = null;
        org.bytedeco.javacv.Java2DFrameConverter converter = new org.bytedeco.javacv.Java2DFrameConverter();
        int gifFps = getGifFps();
        java.io.File tempFile = null;

        try {
            // Get dimensions from first frame
            byte[] firstFrameBytes = frameImages.getFirst();
            BytePointer imagePointer = null;
            Mat encodedFirstFrame = null;
            Mat firstFrame = null;
            try {
                imagePointer = new BytePointer(firstFrameBytes);
                encodedFirstFrame = new org.bytedeco.opencv.opencv_core.Mat(imagePointer);
                firstFrame = org.bytedeco.opencv.global.opencv_imgcodecs.imdecode(
                    encodedFirstFrame,
                    org.bytedeco.opencv.global.opencv_imgcodecs.IMREAD_COLOR
                );

                if (firstFrame.empty()) {
                    log.error("Failed to decode first frame");
                    return null;
                }

                int height = firstFrame.rows();
                int width = firstFrame.cols();

                // Calculate scaled dimensions maintaining aspect ratio
                int scaledWidth = GIF_WIDTH;
                int scaledHeight = (int) ((double) height / width * GIF_WIDTH);

                log.info("Creating MP4 video: {}x{} pixels, {} frames, {} fps",
                    scaledWidth, scaledHeight, frameImages.size(), gifFps);

                // Create temporary file for video
                tempFile = java.io.File.createTempFile("tracking_", ".mp4");
                tempFile.deleteOnExit();

                // Initialize video recorder
                recorder = new org.bytedeco.javacv.FFmpegFrameRecorder(tempFile.getAbsolutePath(), scaledWidth, scaledHeight);
                recorder.setVideoCodec(org.bytedeco.ffmpeg.global.avcodec.AV_CODEC_ID_H264);
                recorder.setFormat("mp4");
                recorder.setFrameRate(gifFps);
                recorder.setPixelFormat(org.bytedeco.ffmpeg.global.avutil.AV_PIX_FMT_YUV420P);
                recorder.setVideoBitrate(2000000); // 2 Mbps
                recorder.start();

                // Process each frame
                for (int i = 0; i < frameImages.size(); i++) {
                    byte[] frameBytes = frameImages.get(i);

                    BytePointer framePointer = null;
                    Mat encodedFrame = null;
                    Mat frameMat = null;
                    Mat resizedFrame = null;
                    Size resizedSize = null;
                    try {
                        // Decode frame
                        framePointer = new BytePointer(frameBytes);
                        encodedFrame = new org.bytedeco.opencv.opencv_core.Mat(framePointer);
                        frameMat = org.bytedeco.opencv.global.opencv_imgcodecs.imdecode(
                            encodedFrame,
                            org.bytedeco.opencv.global.opencv_imgcodecs.IMREAD_COLOR
                        );

                        if (frameMat.empty()) {
                            log.warn("Failed to decode frame {}, skipping", i);
                            continue;
                        }

                        // Resize frame
                        resizedFrame = new Mat();
                        resizedSize = new Size(scaledWidth, scaledHeight);
                        org.bytedeco.opencv.global.opencv_imgproc.resize(frameMat, resizedFrame, resizedSize);

                        // Convert to BufferedImage
                        java.awt.image.BufferedImage bufferedImage = matUtil.matToBufferedImage(resizedFrame);

                        if (bufferedImage != null) {
                            // Convert to Frame and record
                            org.bytedeco.javacv.Frame frame = converter.convert(bufferedImage);
                            recorder.record(frame);
                        }

                        if ((i + 1) % 20 == 0) {
                            log.debug("Processed {}/{} frames", i + 1, frameImages.size());
                        }
                    } finally {
                        if (framePointer != null) {
                            framePointer.deallocate();
                        }
                        if (encodedFrame != null) {
                            encodedFrame.deallocate();
                        }
                        if (resizedSize != null) {
                            resizedSize.deallocate();
                        }
                        matUtil.releaseResources(frameMat, resizedFrame);
                    }
                }

                recorder.stop();
                recorder.release();
                recorder = null;

                // Read the video file into byte array
                byte[] videoBytes = java.nio.file.Files.readAllBytes(tempFile.toPath());
                boolean deleted = tempFile.delete();
                tempFile = null;
                if (!deleted) {
                    log.warn("Failed to delete temporary video file");
                }

                log.info("✅ Successfully created MP4 video: {} bytes", videoBytes.length);
                return videoBytes;
            } finally {
                if (imagePointer != null) {
                    imagePointer.deallocate();
                }
                if (encodedFirstFrame != null) {
                    encodedFirstFrame.deallocate();
                }
                matUtil.releaseResources(firstFrame);
            }

        } catch (Exception e) {
            log.error("Failed to create MP4 video: {}", e.getMessage(), e);
            return null;
        } finally {
            if (recorder != null) {
                try {
                    recorder.stop();
                    recorder.release();
                } catch (Exception e) {
                    log.warn("Error releasing recorder: {}", e.getMessage());
                }
            }
            try {
                converter.close();
            } catch (Exception e) {
                log.warn("Error closing converter: {}", e.getMessage());
            }
            if (tempFile != null && tempFile.exists() && !tempFile.delete()) {
                log.warn("Failed to delete temporary video file: {}", tempFile.getAbsolutePath());
            }
        }
    }

    public int getGifFps() {
        return Math.max(1, faceRecognitionProperties.getTelegram().getGifFps());
    }

    public int getMaxGifFrames() {
        return Math.max(MIN_FRAMES, faceRecognitionProperties.getTelegram().getGifMaxFrames());
    }
}
