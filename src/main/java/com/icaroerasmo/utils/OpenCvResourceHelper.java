package com.icaroerasmo.utils;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

public final class OpenCvResourceHelper {

    private static final String DOCKER_OPENCV_DIRECTORY = "/app/opencv/";

    private OpenCvResourceHelper() {
    }

    /**
     * Get a resource path from Docker filesystem, classpath, or an extracted temp file.
     */
    public static String getResourcePath(String resourceName, Class<?> resourceOwner) throws IOException, URISyntaxException {
        String fileName = resourceName.contains("/")
            ? resourceName.substring(resourceName.lastIndexOf('/') + 1)
            : resourceName;

        File dockerFile = new File(DOCKER_OPENCV_DIRECTORY + fileName);
        if (dockerFile.exists()) {
            return dockerFile.getAbsolutePath();
        }

        var resource = ClassLoader.getSystemResource(resourceName);
        if (resource != null) {
            return Path.of(resource.toURI()).toString();
        }

        try (InputStream inputStream = resourceOwner.getClassLoader().getResourceAsStream(resourceName)) {
            if (inputStream != null) {
                Path tempFile = Files.createTempFile("opencv_", "_" + fileName);
                Files.copy(inputStream, tempFile, StandardCopyOption.REPLACE_EXISTING);
                tempFile.toFile().deleteOnExit();
                return tempFile.toString();
            }
        }

        throw new IOException("Resource not found: " + resourceName
            + ". Checked: " + DOCKER_OPENCV_DIRECTORY + fileName + ", classpath:" + resourceName);
    }
}
