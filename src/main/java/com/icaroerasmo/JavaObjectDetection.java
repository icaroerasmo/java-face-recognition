package com.icaroerasmo;

import com.icaroerasmo.properties.ObjectDetectionProperties;
import com.icaroerasmo.runners.RtspRecognitionRunner;
import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.scheduling.annotation.EnableScheduling;

@Log4j2
@EnableScheduling
@SpringBootApplication
@EnableConfigurationProperties(ObjectDetectionProperties.class)
public class JavaObjectDetection implements CommandLineRunner {

    @Autowired
    private RtspRecognitionRunner rtspRecognitionRunner;

    public static void main(String[] args) {
        SpringApplication.run(JavaObjectDetection.class, args);
    }

    @Override
    @SneakyThrows
    public void run(String... args) {
        rtspRecognitionRunner.start(args);
    }
}
