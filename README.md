# Identify-Small-Objects-in-High-Clutter-Backgrounds
Algorithm for detecting and tracking drones in High Clutter Backgrounds. Uses Unreal Engine 4 to simulate real time camera input. 

## Setup Guide

This project requires **Git LFS** for large files and a **Python Virtual Environment** (`venv`) for dependency management.

### 1. Download Large Files with Git LFS

Large files (like the model weights, e.g., `yolo11n.pt`) are stored outside the main Git repository. You must install the LFS extension to correctly download them.

1.  **Install Git LFS** (if you haven't already):
    ```bash
    git lfs install
    ```
2.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd Identify-Small-Objects-in-High-Clutter-Backgrounds
    ```
3.  **Ensure LFS files are downloaded:** If you cloned before installing LFS, run:
    ```bash
    git lfs pull
    ```

### 2. Configure Python Environment

We use a virtual environment to manage required libraries efficiently using the provided `requirements.txt` file.
This project requires **Python 3.11.9**. Using a different version may cause dependency conflicts due to specific library behaviors (like the YOLO framework) or unexpected runtime issues.

1.  **Create the Virtual Environment:**
    ```bash
    python3 -m venv venv
    ```
2.  **Activate the Environment:**
    * **macOS / Linux:**
        ```bash
        source venv/bin/activate
        ```
    * **Windows (Command Prompt):**
        ```bash
        venv\Scripts\activate.bat
        ```
    *(Your terminal should now show `(venv)`)*

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 3. Execution

1.  **Start UE4 Simulation:** Launch the Unreal Engine project and run the simulation level that outputs the live camera feed.
2.  **Run the Detection Script:** With the Python environment active, execute the main script:
    ```bash
    python Model/Scripts/main_detection_script.py
    ```