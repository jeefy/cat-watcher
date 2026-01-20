# Kubernetes Deployment Guide

This guide covers deploying Cat Watcher to a Kubernetes cluster with GPU support.

## Prerequisites

- Kubernetes cluster (1.24+)
- `kubectl` configured
- NVIDIA GPU Operator (for GPU support)
- Persistent storage provisioner
- Access to container registry

## Quick Start

```bash
# Apply all manifests
kubectl apply -f k8s/

# Or apply individually
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml      # Create this first
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

## Manifest Overview

### Namespace

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: cat-watcher
  labels:
    app.kubernetes.io/name: cat-watcher
```

### ConfigMap

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: cat-watcher-config
  namespace: cat-watcher
data:
  frigate.url: "http://frigate.home.local:5000"
  frigate.camera: "apollo-dish"
  mqtt.host: "mqtt.home.local"
  mqtt.port: "1883"
  inference.confidence_threshold: "0.7"
  inference.use_onnx: "true"
  homeassistant.topic_prefix: "cat_watcher"
```

### Secrets

Create a secrets file (don't commit to git):

```yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: cat-watcher-secrets
  namespace: cat-watcher
type: Opaque
stringData:
  mqtt.username: "cat_watcher"
  mqtt.password: "your-secure-password"
```

Apply secrets:
```bash
kubectl apply -f k8s/secrets.yaml
```

### Persistent Volume Claims

```yaml
# k8s/pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: cat-watcher-data
  namespace: cat-watcher
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: local-path  # Adjust for your cluster
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: cat-watcher-models
  namespace: cat-watcher
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
  storageClassName: local-path
```

### Deployment

The application uses a single unified deployment that handles all functionality (web UI, inference, labeling):

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cat-watcher
  namespace: cat-watcher
  labels:
    app.kubernetes.io/name: cat-watcher
spec:
  replicas: 1
  selector:
    matchLabels:
      app.kubernetes.io/name: cat-watcher
  template:
    metadata:
      labels:
        app.kubernetes.io/name: cat-watcher
    spec:
      initContainers:
        - name: init-permissions
          image: busybox:latest
          command: ['sh', '-c', 'chown -R 1000:1000 /data /models || true']
          volumeMounts:
            - name: data
              mountPath: /data
            - name: models
              mountPath: /models
      containers:
        - name: cat-watcher
          image: jeefy/cat-watcher:latest
          imagePullPolicy: Always
          ports:
            - name: http
              containerPort: 8080
              protocol: TCP
          env:
            - name: CAT_WATCHER__FRIGATE__URL
              valueFrom:
                configMapKeyRef:
                  name: cat-watcher-config
                  key: frigate.url
            - name: CAT_WATCHER__FRIGATE__CAMERA
              valueFrom:
                configMapKeyRef:
                  name: cat-watcher-config
                  key: frigate.camera
            - name: CAT_WATCHER__MQTT__BROKER
              valueFrom:
                configMapKeyRef:
                  name: cat-watcher-config
                  key: mqtt.host
            - name: CAT_WATCHER__MQTT__PORT
              valueFrom:
                configMapKeyRef:
                  name: cat-watcher-config
                  key: mqtt.port
            - name: CAT_WATCHER__MQTT__USERNAME
              valueFrom:
                secretKeyRef:
                  name: cat-watcher-secrets
                  key: mqtt.username
                  optional: true
            - name: CAT_WATCHER__MQTT__PASSWORD
              valueFrom:
                secretKeyRef:
                  name: cat-watcher-secrets
                  key: mqtt.password
                  optional: true
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
              nvidia.com/gpu: "1"
            limits:
              memory: "4Gi"
              cpu: "2000m"
              nvidia.com/gpu: "1"
          volumeMounts:
            - name: data
              mountPath: /data
            - name: models
              mountPath: /models
          livenessProbe:
            httpGet:
              path: /api/health
              port: http
            initialDelaySeconds: 60
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /api/health
              port: http
            initialDelaySeconds: 30
            periodSeconds: 10
      volumes:
        - name: data
          persistentVolumeClaim:
            claimName: cat-watcher-data
        - name: models
          persistentVolumeClaim:
            claimName: cat-watcher-models
      # Optional: Node selector for GPU nodes
      nodeSelector:
        nvidia.com/gpu.present: "true"
      # Optional: Tolerations for GPU nodes
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
```

**Note:** The unified image (`jeefy/cat-watcher:latest`) supports both CPU and CUDA GPU. If running on a CPU-only cluster, remove the `nvidia.com/gpu` resource requests and node selector.

### Service

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: cat-watcher
  namespace: cat-watcher
  labels:
    app.kubernetes.io/name: cat-watcher
spec:
  type: ClusterIP
  ports:
    - port: 8080
      targetPort: http
      protocol: TCP
      name: http
  selector:
    app.kubernetes.io/name: cat-watcher
```

### Ingress (Optional)

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: cat-watcher
  namespace: cat-watcher
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
    - hosts:
        - cat-watcher.example.com
      secretName: cat-watcher-tls
  rules:
    - host: cat-watcher.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: cat-watcher
                port:
                  number: 8000
```

## GPU Configuration

### NVIDIA GPU Operator

Install the NVIDIA GPU Operator for GPU support:

```bash
# Add NVIDIA Helm repo
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

# Install GPU Operator
helm install --wait gpu-operator \
  -n gpu-operator --create-namespace \
  nvidia/gpu-operator
```

### Verify GPU Access

```bash
# Check GPU nodes
kubectl get nodes -l nvidia.com/gpu.present=true

# Check GPU resources
kubectl describe node <gpu-node> | grep -A 10 "Allocatable:"

# Test GPU access
kubectl run gpu-test --rm -it --restart=Never \
  --image=nvidia/cuda:12.0-base \
  --limits=nvidia.com/gpu=1 \
  -- nvidia-smi
```

## Unified Deployment

The cat-watcher project uses a **single unified image** that includes all functionality:
- Web UI
- Inference service  
- Labeling tools
- Data collection

This simplifies deployment and reduces operational complexity. A single deployment handles everything:

```bash
# Deploy the complete application
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

### CPU-Only Clusters

If your cluster doesn't have GPUs, modify the deployment to remove GPU requirements:

```yaml
# Remove from deployment.yaml
resources:
  requests:
    # nvidia.com/gpu: "1"  # Comment out or remove
  limits:
    # nvidia.com/gpu: "1"  # Comment out or remove

# Remove node selector
# nodeSelector:
#   nvidia.com/gpu.present: "true"
```

The unified image automatically falls back to CPU inference when no GPU is available.

## Monitoring

### Prometheus ServiceMonitor

```yaml
# k8s/servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: cat-watcher
  namespace: cat-watcher
  labels:
    app.kubernetes.io/name: cat-watcher
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: cat-watcher
  endpoints:
    - port: http
      path: /metrics
      interval: 30s
```

### Pod Disruption Budget

```yaml
# k8s/pdb.yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: cat-watcher
  namespace: cat-watcher
spec:
  minAvailable: 1
  selector:
    matchLabels:
      app.kubernetes.io/name: cat-watcher
```

## Troubleshooting

### Check Pod Status

```bash
# List pods
kubectl get pods -n cat-watcher

# Describe pod
kubectl describe pod -n cat-watcher <pod-name>

# View logs
kubectl logs -n cat-watcher <pod-name> -f

# Previous container logs (after crash)
kubectl logs -n cat-watcher <pod-name> --previous
```

### Common Issues

**Pod stuck in Pending (GPU)**
```bash
# Check GPU resources
kubectl describe node <node> | grep -A5 "Allocated resources"

# Check events
kubectl get events -n cat-watcher --sort-by='.lastTimestamp'
```

**Image Pull Errors**
```bash
# Check image pull secret
kubectl get secrets -n cat-watcher

# Create registry secret
kubectl create secret docker-registry regcred \
  --docker-server=ghcr.io \
  --docker-username=<username> \
  --docker-password=<token> \
  -n cat-watcher
```

**PVC Pending**
```bash
# Check storage class
kubectl get storageclass

# Check PVC events
kubectl describe pvc -n cat-watcher cat-watcher-data
```

### Debug Shell

```bash
# Start debug pod
kubectl run debug -n cat-watcher --rm -it \
  --image=python:3.11-slim \
  -- bash

# In the pod, test connectivity
pip install httpx
python -c "import httpx; print(httpx.get('http://frigate:5000/api/version').json())"
```

## Upgrades

### Rolling Update

```bash
# Update image
kubectl set image deployment/cat-watcher \
  cat-watcher=jeefy/cat-watcher:v1.2.0 \
  -n cat-watcher

# Watch rollout
kubectl rollout status deployment/cat-watcher -n cat-watcher

# Rollback if needed
kubectl rollout undo deployment/cat-watcher -n cat-watcher
```

### Blue-Green Deployment

```bash
# Deploy new version alongside existing
kubectl apply -f k8s/deployment-v2.yaml

# Switch service to new version
kubectl patch service cat-watcher -n cat-watcher \
  -p '{"spec":{"selector":{"version":"v2"}}}'

# Delete old deployment after verification
kubectl delete deployment cat-watcher-v1 -n cat-watcher
```

## Resource Recommendations

| Configuration | CPU Request | CPU Limit | Memory Request | Memory Limit | GPU |
|---------------|-------------|-----------|----------------|--------------|-----|
| With GPU | 250m | 2000m | 512Mi | 4Gi | 1 |
| CPU Only | 500m | 2000m | 1Gi | 4Gi | - |

Adjust based on your workload and cluster capacity. The unified image handles all functionality (web UI, inference, labeling, collection) in a single deployment.
