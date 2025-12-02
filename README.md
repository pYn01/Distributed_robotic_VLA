Distributed Robotic System with Vision-Language-Action (VLA)

This repository contains the codebase, models, and documentation for a Distributed Robotic System built with ROS 2, Gazebo, and a custom Vision-Language-Action (VLA) architecture.
The system integrates robotic manipulation, distributed visual perception, and neural task planning through action primitives.

🚀 Project Overview
1. Robotic Manipulator

Python-based ROS 2 interface

Kinematic control and trajectory handling

Gazebo/Ignition simulation

High-level action primitives for VLA integration

2. Vision System

Macro detection dataset

YOLO-based training and inference pipeline

Distributed detection node for multi-device processing

3. VLA Architecture

Vision encoder

Language instruction processor

Primitive-based action planner

Supervised learning pipeline + LLM reasoning layer


🛠 Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/DISTRIBUTED_ROBOTIC.git
cd DISTRIBUTED_ROBOTIC


Install all required dependencies (ROS 2, Python packages, YOLO requirements, etc.).

▶️ Usage
1. Start Manipulator Simulation
ros2 launch manipulator_bringup simulation.launch.py

2. Train YOLO Model
python Detection/train.py

3. Run VLA Inference
python VLA/inference/run_vla.py

🧩 Components

Distributed perception: process detection on remote nodes

Robot control stack: position + velocity + primitive-based control

VLA planner: interprets language instructions and executes multi-step actions

Dataset tools: automated macro-detection training and experimentation
