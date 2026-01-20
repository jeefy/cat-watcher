# GPU Slicing Guide

This guide covers configuring GPU sharing and slicing for Cat Watcher on Kubernetes, enabling multiple workloads to share a single GPU.

## Overview

GPU slicing allows you to:
- Run multiple inference services on one GPU
- Share GPU between Cat Watcher and other ML workloads
- Reduce costs by maximizing GPU utilization
- Support smaller GPUs (GTX 1070, 1660, etc.)

## GPU Sharing Technologies

| Technology | NVIDIA Support | Use Case | Isolation |
|------------|---------------|----------|-----------|
| **Time-Slicing** | All GPUs | Basic sharing | None |
| **MPS** | Volta+ | Concurrent kernels | Process |
| **MIG** | A100, A30, H100 | Hardware partitioning | Full |
| **vGPU** | Data center GPUs | Virtual machines | Full |

For home lab setups with consumer GPUs (GTX 1070, 1660, RTX 3060, etc.), **Time-Slicing** and **MPS** are the best options.

## Time-Slicing Configuration

Time-slicing is the simplest approach, allowing multiple pods to share a GPU by time-multiplexing.

### 1. Configure GPU Operator

```yaml
# gpu-operator-values.yaml
devicePlugin:
  config:
    name: time-slicing-config
    default: any
---
# time-slicing-config ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: time-slicing-config
  namespace: gpu-operator
data:
  any: |-
    version: v1
    flags:
      migStrategy: none
    sharing:
      timeSlicing:
        renameByDefault: false
        resources:
          - name: nvidia.com/gpu
            replicas: 4  # Allow 4 pods to share each GPU
```

Apply configuration:

```bash
# Install/upgrade GPU Operator with time-slicing
helm upgrade --install gpu-operator nvidia/gpu-operator \
  -n gpu-operator --create-namespace \
  -f gpu-operator-values.yaml
```

### 2. Verify Time-Slicing

```bash
# Check node capacity (should show 4x GPUs)
kubectl describe node <gpu-node> | grep -A5 "Capacity:"

# Example output:
#   nvidia.com/gpu: 4  # Instead of 1
```

### 3. Deploy with Time-Slicing

```yaml
# k8s/deployment-timeslice.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cat-watcher
  namespace: cat-watcher
spec:
  replicas: 1
  selector:
    matchLabels:
      app.kubernetes.io/name: cat-watcher
  template:
    spec:
      containers:
        - name: cat-watcher
          image: jeefy/cat-watcher:latest
          resources:
            limits:
              nvidia.com/gpu: 1  # Gets 1/4 of GPU time
```

### Time-Slicing Considerations

| Pros | Cons |
|------|------|
| Simple to configure | No memory isolation |
| Works on all GPUs | Context switching overhead |
| No code changes needed | GPU memory shared |
| Kubernetes native | Potential OOM if overcommitted |

## MPS (Multi-Process Service)

MPS provides better concurrency by allowing CUDA kernels from different processes to run simultaneously.

### 1. Enable MPS in GPU Operator

```yaml
# mps-config ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: mps-config
  namespace: gpu-operator
data:
  any: |-
    version: v1
    flags:
      migStrategy: none
    sharing:
      mps:
        resources:
          - name: nvidia.com/gpu
            replicas: 4
            # Memory limit per process (optional)
            # memoryGB: 2
```

### 2. MPS Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cat-watcher
  namespace: cat-watcher
spec:
  template:
    spec:
      containers:
        - name: cat-watcher
          image: jeefy/cat-watcher:latest
          env:
            # Enable MPS client
            - name: CUDA_MPS_PIPE_DIRECTORY
              value: /tmp/nvidia-mps
            - name: CUDA_MPS_LOG_DIRECTORY
              value: /tmp/nvidia-mps-log
          resources:
            limits:
              nvidia.com/gpu: 1
          volumeMounts:
            - name: mps-pipe
              mountPath: /tmp/nvidia-mps
      volumes:
        - name: mps-pipe
          hostPath:
            path: /tmp/nvidia-mps
            type: DirectoryOrCreate
```

### MPS Considerations

| Pros | Cons |
|------|------|
| Better GPU utilization | Requires Volta+ (some features) |
| Concurrent kernel execution | Shared failure domain |
| Lower latency | More complex setup |
| Memory limits possible | Not all CUDA features supported |

## MIG (Multi-Instance GPU)

MIG provides hardware-level partitioning for A100, A30, and H100 GPUs.

### 1. Enable MIG

```bash
# On the GPU node
sudo nvidia-smi mig 1

# Create MIG instances (A100 40GB example)
# 7 instances of 1g.5gb each
sudo nvidia-smi mig -cgi 19,19,19,19,19,19,19 -C

# Or larger instances
# 2 instances of 3g.20gb
sudo nvidia-smi mig -cgi 9,9 -C
```

### 2. GPU Operator MIG Configuration

```yaml
# mig-config ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: mig-config
  namespace: gpu-operator
data:
  config.yaml: |-
    version: v1
    mig-configs:
      all-1g.5gb:
        - devices: all
          mig-enabled: true
          mig-devices:
            "1g.5gb": 7
      all-3g.20gb:
        - devices: all
          mig-enabled: true
          mig-devices:
            "3g.20gb": 2
```

### 3. MIG Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cat-watcher
  namespace: cat-watcher
spec:
  template:
    spec:
      containers:
        - name: cat-watcher
          image: jeefy/cat-watcher:latest
          resources:
            limits:
              # Request specific MIG profile
              nvidia.com/mig-1g.5gb: 1
```

## Memory Management

### Limiting GPU Memory (ONNX Runtime)

Configure Cat Watcher to limit GPU memory usage:

```python
# In inference service
import onnxruntime as ort

# Limit to 2GB
sess_options = ort.SessionOptions()
cuda_options = {
    "device_id": 0,
    "arena_extend_strategy": "kSameAsRequested",
    "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2GB
}

session = ort.InferenceSession(
    "model.onnx",
    sess_options,
    providers=[("CUDAExecutionProvider", cuda_options)],
)
```

### Environment Variable Configuration

```yaml
# deployment.yaml
env:
  - name: CAT_WATCHER__INFERENCE__GPU_MEMORY_LIMIT
    value: "2048"  # MB
  - name: CUDA_VISIBLE_DEVICES
    value: "0"
  - name: TF_FORCE_GPU_ALLOW_GROWTH
    value: "true"
```

## Recommended Configurations

### GTX 1070 (8GB)

```yaml
# 2 inference pods sharing GPU
sharing:
  timeSlicing:
    resources:
      - name: nvidia.com/gpu
        replicas: 2

# Per-pod memory limit: ~3.5GB
# Reserve ~1GB for system
```

### GTX 1660 (6GB)

```yaml
# 2 inference pods, memory constrained
sharing:
  timeSlicing:
    resources:
      - name: nvidia.com/gpu
        replicas: 2

# Per-pod memory limit: ~2.5GB
# Use smaller models (YOLOv8n, EfficientNet-B0)
```

### RTX 3060 (12GB)

```yaml
# 4 inference pods
sharing:
  timeSlicing:
    resources:
      - name: nvidia.com/gpu
        replicas: 4

# Per-pod memory limit: ~2.5GB
# Can use larger models
```

### RTX 3090/4090 (24GB)

```yaml
# 8+ inference pods with MPS
sharing:
  mps:
    resources:
      - name: nvidia.com/gpu
        replicas: 8

# Per-pod memory limit: ~2.5GB
# Plenty of headroom for larger models
```

## Monitoring GPU Usage

### NVIDIA DCGM Exporter

```bash
# Install DCGM Exporter
helm install dcgm-exporter nvidia/dcgm-exporter \
  -n monitoring --create-namespace
```

### Prometheus Queries

```promql
# GPU utilization per pod
DCGM_FI_DEV_GPU_UTIL{pod=~"cat-watcher.*"}

# GPU memory used
DCGM_FI_DEV_FB_USED{pod=~"cat-watcher.*"}

# GPU memory free
DCGM_FI_DEV_FB_FREE{pod=~"cat-watcher.*"}
```

### Grafana Dashboard

Import NVIDIA DCGM dashboard (ID: 12239) for comprehensive GPU monitoring.

## Troubleshooting

### Out of GPU Memory

```bash
# Check GPU memory
nvidia-smi

# In pod
python -c "import torch; print(torch.cuda.memory_summary())"
```

Solutions:
- Reduce batch size
- Use smaller models
- Reduce time-slice replicas
- Set memory limits in ONNX config

### Time-Slicing Contention

```bash
# Monitor GPU utilization
watch -n1 nvidia-smi

# If utilization is 100% constantly, reduce replicas
```

### MPS Failures

```bash
# Check MPS daemon
nvidia-cuda-mps-control -s

# Restart MPS
echo quit | nvidia-cuda-mps-control
nvidia-cuda-mps-control -d
```

## Best Practices

1. **Start Conservative**: Begin with 2 replicas and increase
2. **Monitor Memory**: Track GPU memory usage over time
3. **Use ONNX**: ONNX Runtime has better memory management
4. **Set Limits**: Always configure memory limits
5. **Test Thoroughly**: Verify performance under load
6. **Graceful Degradation**: Implement CPU fallback

## Example: Complete Time-Slicing Setup

```bash
# 1. Configure GPU Operator
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: time-slicing-config
  namespace: gpu-operator
data:
  any: |-
    version: v1
    sharing:
      timeSlicing:
        resources:
          - name: nvidia.com/gpu
            replicas: 4
EOF

# 2. Restart device plugin
kubectl rollout restart daemonset/nvidia-device-plugin-daemonset -n gpu-operator

# 3. Verify
kubectl describe node <gpu-node> | grep nvidia.com/gpu

# 4. Deploy Cat Watcher
kubectl apply -f k8s/deployment.yaml

# 5. Scale up (will share GPU)
kubectl scale deployment cat-watcher-inference --replicas=4 -n cat-watcher
```

This allows 4 Cat Watcher pods to share a single GPU with time-slicing!
