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
