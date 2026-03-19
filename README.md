# GrokPendulum: Phase Transitions in Physical Representation Learning

## Project Context
This project investigates the **Grokking** (delayed generalization) phenomenon within Masked Autoencoders (MAEs) trained on a synthetic physics dataset. While foundational models rely on diverse, real-world data, this project isolates physical dynamics by utilizing a perfectly controlled Unity 3D environment. 

The objective is to observe if and how a model transitions from *memorizing* the dataset to *understanding* the underlying symbolic physics rules (e.g., pendulum dynamics and camera ego-motion) after extended training well beyond the point of overfitting.

## Architecture & Data Pipeline
The project is divided into two decoupled systems:

1. **The Physics Engine (Unity C#):** A strict, gravity-driven pendulum system built in Unity (URP). It autonomously randomizes physical properties (initial angle, damping/friction mapped to colors) and camera ego-motion, generating highly synchronized $(S_t, A_t) \rightarrow S_{t+1}$ state transitions.
2. **The Representation Learner (PyTorch):** A custom Vision Transformer (ViT) based Masked Autoencoder designed to process $64 \times 64$ resolution image pairs, bottlenecking the information to force the learning of physical parameters.

## Dataset Structure
The dataset consists of `.png` frames and a corresponding `ground_truth.csv` file.
* **State ($S_t$):** $64 \times 64$ RGB image of the pendulum.
* **Action ($A_t$):** Camera movement (Delta Azimuth/Elevation).
* **Target ($S_{t+1}$):** The subsequent physical frame.
* **Hidden Variables:** Damping factor (mapped to bob color), pendulum angle, and exact camera coordinates.