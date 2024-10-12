# real_time_video_analytics
Real Time Video Content Analysis for Construction

![image](https://github.com/user-attachments/assets/bbe0d4e3-30d4-47ff-8eaf-9ffeb82c90eb)

# Real-Time Video Analytics for Construction Safety

This Python application leverages a retrained YOLOv11 model on a publicly available construction site safety dataset to detect unsafe practices in video footage. A sample video file has been provided for testing, showcasing recut clips of popular YouTube fails.

## Dataset & Video Source

- **Dataset**: [Construction Site Safety](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety)
- **Video Source**: [YouTube Fails](https://www.youtube.com/watch?v=FI1XrdBJIUI)

The video file `unsafe_practices.mov` is located in the `data/` folder for testing purposes.

## Features

- **Model Training**: The `training_custom_dataset.ipynb` notebook can be used to retrain the YOLOv11 instance on the construction safety dataset. The retrained weights will be saved in the `runs/train_x/weights/` directory.
- **Video Inference**: Run real-time video analytics to detect unsafe practices in the provided video or any custom video.

## Getting Started

### Prerequisites

- Python >= 3.7
- Virtual environment tool (such as `venv`)

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-repository-url.git
   cd /Users/noah_/Documents/Development/2024_Projects/real_time_video_analytics

2. Create and Activate a Virtual Environment
```bash

python3 -m venv /Users/noah_/Documents/Development/2024_Projects/real_time_video_analytics/groundup
source /Users/noah_/Documents/Development/2024_Projects/real_time_video_analytics/groundup/bin/activate

```

3. Install Requirements
```bash
pip install -r requirements.txt
```

4. Running the Application
To run the video analytics application with the provided unsafe_practices.mov file, use the following commands:

```bash
cd /{YOUR_SOURCE}/real_time_video_analytics
source /{YOUR_SOURCE}/real_time_video_analytics/groundup/bin/activate
python -m app --video_file_path=data/unsafe_practices.mov

```

5. Retraining the Model
If you want to retrain the YOLOv11 model on the dataset:

Open the training_custom_dataset.ipynb notebook.
Follow the instructions within the notebook to retrain the model.
The weights will be saved in the runs/train_x/weights/ directory upon completion.


6. Folder Structure:
├── app/
│   ├── __init__.py
│   ├── app.py
│   └── utils.py
├── data/
│   └── unsafe_practices.mov
├── runs/
│   └── train_x/
│       └── weights/
├── training_custom_dataset.ipynb
├── requirements.txt
└── README.md


7. License
This project is licensed under the MIT License - see the LICENSE file for details.

This `README.md` provides clear instructions for users to set up and run the application, including information on how to retrain the model. Let me know if you'd like to adjust any part!
