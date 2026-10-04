# Live runtime profiles (educational pilot)

Live uses the same YOLO and X3D weights as Upload & Analyze. No model is retrained.

## CPU-only

Create a Python 3.11 environment and install `requirements.txt`. Choose **CPU** or **Auto** in the UI. The current OpenCV YuNet face detector runs on CPU for every received frame. CPU is supported for Upload & Analyze; Live can enter `degraded` if its bounded in-memory queue fills. No raw-frame queue is written to disk.

## NVIDIA CUDA

Install a CUDA-enabled PyTorch wheel and an OpenCV build compiled with CUDA that match the machine's NVIDIA driver. CUDA selection uses OpenCV `FaceDetectorYN`'s DNN CUDA backend for YuNet plus PyTorch CUDA for YOLO/X3D. OpenCV exposes backend/target selection in its API; standard `opencv-python` wheels may not include CUDA. ONNX Runtime providers are reported for diagnostics; YuNet currently uses OpenCV DNN rather than ONNX Runtime. The ONNX Runtime CUDA compatibility matrix changes over time; see the official [ONNX Runtime CUDA Execution Provider guide](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) if adding ORT workloads later. **Auto selects CUDA only when both PyTorch CUDA and an operational CUDA-enabled OpenCV YuNet runtime are detected; otherwise it uses CPU.** Explicit CUDA fails with a clear runtime error if the complete CUDA path is unavailable.

Check `/api/health` for PyTorch CUDA availability, GPU name/VRAM, ONNX providers and the YuNet backend. Do not infer a 2-second or 25 FPS guarantee from the presence of a GPU. Benchmark a machine/source profile first; if Live processing falls behind, the server stops capture, drains already-received frames and reports `degraded`.

RTSP streams and replay files are processed in memory. For replay, provide a video path under the repository's `data/` directory. RTSP URLs are not returned by the API and must not be copied into logs or screenshots because they may contain credentials. The pilot API has no user authentication; bind it to loopback unless access controls are added.
