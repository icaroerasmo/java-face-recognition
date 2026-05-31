# GPU acceleration

This project can use the GPU **now** through **OpenCL** for the OpenCV DNN-based detectors.

Current status:

- **Works today:** OpenCV DNN on **OpenCL**
- **Does not work today:** OpenCV DNN on **CUDA**
- **Still CPU-only:** `LBPHFaceRecognizer`

Why:

- The app uses JavaCV / Bytedeco OpenCV.
- The packaged OpenCV build in that stack does **not** include CUDA DNN support.
- OpenCL is available and can be used if the host and container expose the NVIDIA OpenCL runtime correctly.

## What was added in this branch

- Configurable acceleration settings in `config.yaml`
- Automatic OpenCV acceleration selection with CPU fallback
- Per-thread face/person detector nets instead of one shared synchronized net
- Docker image support for OpenCL loader and Java native access

## 1. Host requirements

You need:

- NVIDIA GPU
- NVIDIA driver working on the host
- `nvidia-container-toolkit`
- `ocl-icd`
- `opencl-nvidia`

On Arch Linux:

```bash
sudo pacman -S --noconfirm nvidia-utils nvidia-container-toolkit ocl-icd opencl-nvidia
```

Optional 32-bit package:

```bash
sudo pacman -S --noconfirm lib32-opencl-nvidia
```

## 2. Verify the host GPU stack

Check NVIDIA:

```bash
nvidia-smi
```

Check OpenCL:

```bash
clinfo
```

Expected result:

- a platform named something like `NVIDIA CUDA`
- the GPU listed as an OpenCL device

Also confirm the ICD file exists:

```bash
ls -l /etc/OpenCL/vendors/nvidia.icd
```

## 3. Project config

In `src/main/resources/config.yaml` or your runtime config, use:

```yaml
face-recognition:
  acceleration:
    backend: auto
    target: auto
    face-detection-target: auto
    person-detection-target: cpu
    enable-opencl: true
    fallback-to-cpu: true
```

### Acceleration settings

`backend`:

- `auto`
- `default`
- `opencv`
- `cuda`

`target`:

- `auto`
- `cpu`
- `opencl`
- `opencl_fp16`
- `cuda`
- `cuda_fp16`

Recommended today:

```yaml
backend: auto
target: auto
face-detection-target: auto
person-detection-target: cpu
enable-opencl: true
fallback-to-cpu: true
```

If you want to force OpenCL:

```yaml
backend: opencv
target: opencl
enable-opencl: true
fallback-to-cpu: false
```

## 4. Build the image

From the repository root:

```bash
mvn clean package
docker build -t rtsp-face-recognition:feature-gpu-acceleration .
```

## 5. Run the container with GPU access

Example Docker Compose service:

```yaml
rtsp-face-recognition:
  image: rtsp-face-recognition:feature-gpu-acceleration
  container_name: rtsp-face-recognition
  restart: unless-stopped
  runtime: nvidia
  environment:
    TZ: America/Bahia
    TELEGRAM_CHAT_ID: ${TELEGRAM_CHAT_ID}
    TELEGRAM_BOT_TOKEN: ${TELEGRAM_BOT_TOKEN}
    NVIDIA_VISIBLE_DEVICES: ${NVIDIA_VISIBLE_DEVICES:-all}
    NVIDIA_DRIVER_CAPABILITIES: ${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}
  volumes:
    - ./config:/app/config:ro
    - ./train:/app/train:ro
    - ./data:/app/data
```

Then start it:

```bash
docker compose up -d rtsp-face-recognition
```

## 6. Verify GPU acceleration in the container

Check NVIDIA visibility:

```bash
docker exec rtsp-face-recognition nvidia-smi
```

Check OpenCL libraries:

```bash
docker exec rtsp-face-recognition sh -lc 'ls -l /etc/OpenCL/vendors && ldconfig -p | grep -Ei "OpenCL|nvidia-opencl"'
```

Check application logs:

```bash
docker logs rtsp-face-recognition | grep "OpenCV acceleration initialized"
```

Expected GPU-enabled log:

```text
OpenCV acceleration initialized: requested backend=AUTO, requested target=AUTO, openclEnabled=true, openclAvailable=true, cudaSupported=false, fallbackToCpu=true
```

If OpenCL is actually selected, detector configuration logs should show:

```text
Configured face detection net with backend=OPENCV and target=OPENCL
Configured person detection net with backend=OPENCV and target=CPU
```

That split is intentional: person detection stays on CPU because OpenCL on the SSD MobileNet path caused missed detections in live use, while face detection can still use OpenCL safely.

## 7. Troubleshooting

### `openclAvailable=false`

Usually means one of these:

- `opencl-nvidia` is missing on the host
- `/etc/OpenCL/vendors/nvidia.icd` does not exist on the host
- container started without NVIDIA runtime
- NVIDIA driver/container runtime is not exposing the OpenCL libraries

Check:

```bash
clinfo
docker exec rtsp-face-recognition nvidia-smi
docker exec rtsp-face-recognition sh -lc 'ldconfig -p | grep -Ei "OpenCL|nvidia-opencl"'
```

### `cudaSupported=false`

This is expected with the current JavaCV / Bytedeco OpenCV native build.

The current packaged OpenCV binaries do not include CUDA modules, so this project cannot use:

- `DNN_BACKEND_CUDA`
- `DNN_TARGET_CUDA`
- `DNN_TARGET_CUDA_FP16`

### The app still uses CPU for recognition

This is also expected.

`LBPHFaceRecognizer` is CPU-only. In the current architecture, only the DNN-based:

- face detection
- person detection

can move to GPU.

## 8. What is required for real CUDA acceleration

To use CUDA instead of OpenCL, one of these bigger changes is required:

1. Build and ship a **custom JavaCPP / OpenCV preset** with CUDA-enabled OpenCV
2. Replace the detector stack with a GPU-native runtime such as:
   - ONNX Runtime GPU
   - TensorRT
   - another CUDA-capable inference engine

For full GPU recognition, the LBPH recognizer would also need to be replaced with an embedding-based model.

## 9. Recommended production setting today

Use:

```yaml
face-recognition:
  acceleration:
    backend: auto
    target: auto
    enable-opencl: true
    fallback-to-cpu: true
```

That gives:

- OpenCL GPU when available
- safe CPU fallback when not available
