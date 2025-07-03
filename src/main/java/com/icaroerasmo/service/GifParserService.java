package com.icaroerasmo.service;

import com.icaroerasmo.model.GifFrame;
import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.apache.commons.collections4.QueueUtils;
import org.apache.commons.collections4.queue.CircularFifoQueue;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.springframework.stereotype.Service;

import javax.imageio.*;
import javax.imageio.metadata.IIOInvalidTreeException;
import javax.imageio.metadata.IIOMetadata;
import javax.imageio.metadata.IIOMetadataNode;
import javax.imageio.stream.FileImageOutputStream;
import javax.imageio.stream.ImageOutputStream;
import java.awt.image.BufferedImage;
import java.awt.image.RenderedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicInteger;

@Log4j2
@Service
public class GifParserService {

    private static AtomicInteger count = new AtomicInteger(0);

    public static final int BUFFER_SIZE = 100;
    private static final Queue<GifFrame> GIF_BUFFER = QueueUtils.synchronizedQueue(new CircularFifoQueue<>(BUFFER_SIZE));

    public void generateGif() {
        synchronized (GIF_BUFFER) {
            GIF_BUFFER.notifyAll();
        }
    }

    @SneakyThrows
    public void addFrame(Mat img) {
        synchronized (GIF_BUFFER) {
            GIF_BUFFER.add(new GifFrame(Instant.now(), img));
        }
    }

    private List<Mat> getFrames() {
        LinkedList<Mat> frames = new LinkedList<>();
        while (!GIF_BUFFER.isEmpty()) {
            GifFrame frame = GIF_BUFFER.poll();
            if (frame != null) {
                frames.add(frame.getFrame());
            }
        }
        return frames;
    }

    /**
     * Inicia o processo de criação do GIF em segundo plano.
     * Este método executa um loop que verifica periodicamente o buffer de frames
     * e cria um GIF quando há frames suficientes.
     */
    private void runGifCreator() {
        CompletableFuture.runAsync(() -> {
            while(true) {
                try {

                    synchronized (GIF_BUFFER) {

                        GIF_BUFFER.wait();

                        if(GIF_BUFFER.size() < BUFFER_SIZE) {
                            continue;
                        }

                        List<Mat> gifFrames = new ArrayList<Mat>(getFrames());

                        createGif(gifFrames, "output_"+count.getAndIncrement()+".gif");
                    }

                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt(); // Restore interrupted status
                    break; // Exit the loop if interrupted
                }
            }
        });
    }

    public void createGif(List<Mat> images, String outputPath) {
        try {
            BufferedImage[] bufferedImages = new BufferedImage[images.size()];

            // Converte cada Mat para BufferedImage com redimensionamento
            for (int i = 0; i < images.size(); i++) {
                Mat mat = images.get(i);

                // Calcula o novo tamanho mantendo a proporção
                Mat resized = new Mat();
                double aspectRatio = (double) mat.cols() / mat.rows();
                int targetHeight = 480;
                int targetWidth = (int) (targetHeight * aspectRatio);

                // Se a largura for maior que 640, ajusta baseado na largura
                if (targetWidth > 640) {
                    targetWidth = 640;
                    targetHeight = (int) (targetWidth / aspectRatio);
                }

                opencv_imgproc.resize(mat, resized, new Size(targetWidth, targetHeight));

                Java2DFrameConverter converter = new Java2DFrameConverter();
                OpenCVFrameConverter.ToMat matConverter = new OpenCVFrameConverter.ToMat();
                Frame frame = matConverter.convert(resized);
                BufferedImage originalBI = converter.getBufferedImage(frame);

                // Reduz a qualidade da imagem
                bufferedImages[i] = reduceQuality(originalBI, 0.5f);

                // Libera a Mat redimensionada
                resized.release();
            }

            // Cria o GIF com menos frames por segundo (aumenta o delay entre frames)
            ImageOutputStream output = new FileImageOutputStream(new File(outputPath));
            GifSequenceWriter writer = new GifSequenceWriter(output, bufferedImages[0].getType(), 100, true);

            // Adiciona cada frame ao GIF
            for (BufferedImage image : bufferedImages) {
                writer.writeToSequence(image);
            }

            writer.close();
            output.close();
        } catch (IOException e) {
            log.error("Erro ao criar GIF", e);
            throw new RuntimeException("Falha ao criar GIF", e);
        } finally {
            // Libera os recursos
            for (Mat mat : images) {
                if (mat != null && !mat.isNull()) {
                    mat.release();
                }
            }
        }
    }

    private BufferedImage reduceQuality(BufferedImage source, float quality) {
        try {
            // Cria um ByteArrayOutputStream para armazenar a imagem comprimida
            ByteArrayOutputStream baos = new ByteArrayOutputStream();

            // Configura os parâmetros de compressão JPEG
            Iterator<ImageWriter> writers = ImageIO.getImageWritersByFormatName("jpg");
            ImageWriter writer = writers.next();
            ImageWriteParam param = writer.getDefaultWriteParam();
            param.setCompressionMode(ImageWriteParam.MODE_EXPLICIT);
            param.setCompressionQuality(quality);

            // Comprime a imagem
            ImageOutputStream ios = ImageIO.createImageOutputStream(baos);
            writer.setOutput(ios);
            writer.write(null, new IIOImage(source, null, null), param);
            writer.dispose();

            // Converte de volta para BufferedImage
            byte[] bytes = baos.toByteArray();
            return ImageIO.read(new ByteArrayInputStream(bytes));
        } catch (IOException e) {
            log.error("Erro ao reduzir qualidade da imagem", e);
            return source;
        }
    }

    // Classe auxiliar para escrever o GIF
    private class GifSequenceWriter {
        private ImageWriter writer;
        private ImageWriteParam params;
        private IIOMetadata metadata;

        public GifSequenceWriter(ImageOutputStream out, int imageType, int delayTime, boolean loop) throws IOException {
            writer = ImageIO.getImageWritersBySuffix("gif").next();
            params = writer.getDefaultWriteParam();

            ImageTypeSpecifier imageTypeSpecifier = ImageTypeSpecifier.createFromBufferedImageType(imageType);
            metadata = writer.getDefaultImageMetadata(imageTypeSpecifier, params);

            configureGifMetadata(delayTime, loop);
            writer.setOutput(out);
            writer.prepareWriteSequence(null);
        }

        private void configureGifMetadata(int delayTime, boolean loop) throws IIOInvalidTreeException {
            String metaFormatName = metadata.getNativeMetadataFormatName();
            IIOMetadataNode root = (IIOMetadataNode) metadata.getAsTree(metaFormatName);

            IIOMetadataNode graphicsControlExtensionNode = getNode(root, "GraphicControlExtension");
            graphicsControlExtensionNode.setAttribute("disposalMethod", "none");
            graphicsControlExtensionNode.setAttribute("userInputFlag", "FALSE");
            graphicsControlExtensionNode.setAttribute("transparentColorFlag", "FALSE");
            graphicsControlExtensionNode.setAttribute("delayTime", Integer.toString(delayTime / 10));
            graphicsControlExtensionNode.setAttribute("transparentColorIndex", "0");

            IIOMetadataNode commentsNode = getNode(root, "CommentExtensions");
            commentsNode.setAttribute("CommentExtension", "Created by JavaCV");

            IIOMetadataNode appExtensionsNode = getNode(root, "ApplicationExtensions");
            IIOMetadataNode child = new IIOMetadataNode("ApplicationExtension");
            child.setAttribute("applicationID", "NETSCAPE");
            child.setAttribute("authenticationCode", "2.0");
            child.setUserObject(new byte[]{1, (byte) (loop ? 0 : 1), 0});
            appExtensionsNode.appendChild(child);

            metadata.setFromTree(metaFormatName, root);
        }

        private static IIOMetadataNode getNode(IIOMetadataNode rootNode, String nodeName) {
            for (int i = 0; i < rootNode.getLength(); i++) {
                if (rootNode.item(i).getNodeName().equalsIgnoreCase(nodeName)) {
                    return (IIOMetadataNode) rootNode.item(i);
                }
            }
            IIOMetadataNode node = new IIOMetadataNode(nodeName);
            rootNode.appendChild(node);
            return node;
        }

        public void writeToSequence(RenderedImage img) throws IOException {
            writer.writeToSequence(new IIOImage(img, null, metadata), params);
        }

        public void close() throws IOException {
            writer.endWriteSequence();
            writer.dispose();
        }
    }
}
