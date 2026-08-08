package com.icaroerasmo.utils;

import javax.imageio.ImageIO;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;

public final class FaceHashUtils {

    private static final int HASH_WIDTH = 9;
    private static final int HASH_HEIGHT = 8;

    private FaceHashUtils() {
    }

    public static byte[] computePerceptualHash(byte[] encodedImage) throws IOException {
        BufferedImage source = ImageIO.read(new ByteArrayInputStream(encodedImage));
        if (source == null) {
            throw new IOException("Unable to decode image for perceptual hashing");
        }

        BufferedImage scaled = new BufferedImage(HASH_WIDTH, HASH_HEIGHT, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D graphics = scaled.createGraphics();
        try {
            graphics.setRenderingHint(
                RenderingHints.KEY_INTERPOLATION,
                RenderingHints.VALUE_INTERPOLATION_BILINEAR
            );
            graphics.drawImage(source, 0, 0, HASH_WIDTH, HASH_HEIGHT, null);
        } finally {
            graphics.dispose();
        }

        byte[] hash = new byte[HASH_HEIGHT];
        for (int y = 0; y < HASH_HEIGHT; y++) {
            int row = 0;
            for (int x = 0; x < HASH_WIDTH - 1; x++) {
                int current = scaled.getRaster().getSample(x, y, 0);
                int next = scaled.getRaster().getSample(x + 1, y, 0);
                if (current > next) {
                    row |= 1 << x;
                }
            }
            hash[y] = (byte) row;
        }
        return hash;
    }

    /**
     * Compute the normalized Hamming distance between two perceptual hashes.
     * Returns a score from 0 (identical) to 100 (completely different).
     */
    public static int computeSimilarity(byte[] hash1, byte[] hash2) {
        if (hash1 == null || hash2 == null || hash1.length == 0 || hash1.length != hash2.length) {
            return 100;
        }

        int differentBits = 0;
        for (int i = 0; i < hash1.length; i++) {
            differentBits += Integer.bitCount((hash1[i] ^ hash2[i]) & 0xff);
        }

        return (differentBits * 100) / (hash1.length * Byte.SIZE);
    }
}
