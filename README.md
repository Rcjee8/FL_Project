Project Description

This project implements a privacy-presing face recognition system using Federated Learning (FL) combined with Differential Privacy (DP).

The system enables multiple edge devices (Android phones) to collaboratively train a shared model without transmitting raw biometric data. Instead, only model weight updates are communicated to a central server.

Key Features:
Federated Learning using FedAvg
Differential Privacy using DP-SGD
Zero-install mobile client via browser (TensorFlow.js)
Offline-capable inference system
Real-time face detection using BlazeFace
Secure communication over HTTPS (local network)
Platform:
Server: Flask (Python)
Client: Browser-based (JavaScript + TensorFlow.js)
Dataset: Labeled Faces in the Wild (LFW)

🔹 1. Zero-install Federated Learning Client
Runs entirely in browser (no APK)
Uses TensorFlow.js
🔹 2. Fully Offline FL System
Works over local WiFi
No internet required after setup
🔹 3. Differential Privacy Integration
Gradient clipping (C = 1.0)
Gaussian noise (σ = 0.02)
🔹 4. Fix for TF.js Android Bug
Replaced Adam optimizer with custom SGD
Prevents WebGL crash
🔹 5. Supplement Dataset Mechanism
Improves non-IID performance
Adds balanced class samples
Experiment Details
Dataset:
LFW (filtered to ~15 identities)
Training Setup:
FL rounds: 10
Local epochs: 3
Batch size: 4 (phone), 8 (server)
Learning rate: 0.005
Results:
Metric	Value
Accuracy (with DP)	84%
Accuracy (without DP)	89%
Privacy loss (ε)	~4.2
FPS	~12
Observations:
DP introduces ~5% accuracy drop
Supplement data improves accuracy significantly
System scales with multiple clients
