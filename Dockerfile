FROM maven:3.8.8-amazoncorretto-21 AS build
WORKDIR /app
COPY pom.xml .
COPY src ./src/
RUN mvn clean package

FROM eclipse-temurin:21-jre-jammy
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Update package lists and install required native dependencies for OpenCV/JavaCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    rclone \
    tzdata \
    libgtk-3-0 \
    libgtk2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libx11-6 \
    libfontconfig1 \
    libice6 \
    libgomp1 \
    libquadmath0 \
    libgfortran5 \
    libstdc++6 \
    libopenblas0 \
    libopenblas-dev \
    liblapack3 \
    liblapack-dev \
    libblas3 \
    libjpeg-turbo-progs \
    libpng-dev \
    libtiff5 \
    libpython3.10 \
    && rm -rf /var/lib/apt/lists/*

# Set timezone
ARG TZ=UTC
ENV TZ=${TZ}
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Create required directories
RUN mkdir -p /app/data/tmp /app/data/records /app/train /app/config /app/opencv

# Copy the built jar from the build stage
COPY --from=build /app/target/rtsp-object-detection-*.jar /app/rtsp-object-detection.jar

# Copy training data from build stage if it exists
RUN cp -r /app/target/classes/train/* /app/train/ 2>/dev/null || echo "Training data not found in build stage"

# Copy OpenCV model files from build stage
COPY --from=build /app/target/classes/opencv/ /app/opencv/

RUN ls -la /app && ls -la /app/train 2>/dev/null || echo "train directory may be empty" && ls -la /app/opencv 2>/dev/null || echo "opencv directory may be empty"

ENTRYPOINT [ "java", "-Dspring.config.additional-location=/app/config/config.yaml", "-jar", "/app/rtsp-object-detection.jar" ]
