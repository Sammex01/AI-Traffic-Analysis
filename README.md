# 👁️ AI Traffic Analysis: Computer Vision to Exploratory Data Analysis (EDA)

### 🎯 Objective
City planners and logistics companies need to understand traffic volume, but manually counting vehicles and pedestrians from video footage is highly inefficient. The goal of this project was to build an automated AI pipeline that processes raw, unstructured traffic camera footage, detects and tracks individual entities in real-time, and converts that visual data into a structured dataset for Exploratory Data Analysis (EDA).

### 🛠️ Methodology & Technical Stack
This project bridges the gap between deep learning model inference and traditional data analysis.

* **Computer Vision (AI):** Deployed the pre-trained **YOLOv8** (You Only Look Once) deep learning model via `ultralytics` to perform pixel-perfect object detection and classification (cars, pedestrians, buses) on a frame-by-frame basis.
* **Video Processing:** Utilized `OpenCV` to handle video stream extraction and frame manipulation.
* **Data Engineering:** Engineered a custom extraction loop to pull raw tensor data (bounding box coordinates, class IDs, tracking IDs, and confidence scores) directly from the AI's memory and flatten it into a structured `Pandas` DataFrame.
* **Exploratory Data Analysis (EDA):** Cleaned the resulting dataset by removing duplicate entity IDs across multiple frames to calculate the true volume of unique objects in the footage, visualizing the findings using `Seaborn` and `Matplotlib`.

### 🧠 Code Glimpse: Video to Database Conversion
Here is the core logic used to intercept the AI's tracking math and convert it into a tabular dataset:

```python
# Run YOLO with object tracking enabled
results = model.track(frame, persist=True, verbose=False)

# Intercept the raw data from the model's memory
if results[0].boxes.id is not None:
    boxes = results[0].boxes.xyxy.cpu().numpy() # Bounding box coordinates
    track_ids = results[0].boxes.id.cpu().numpy() # Unique entity ID
    classes = results[0].boxes.cls.cpu().numpy() # Object category
    
    # Loop through every object in the frame and append to our database
    for box, track_id, cls in zip(boxes, track_ids, classes):
        traffic_data.append({
            'Frame': frame_count,
            'Object_ID': int(track_id),
            'Class_Name': model.names[int(cls)], 
            'X_Center': (box[0] + box[2]) / 2, 
            'Y_Center': (box[1] + box[3]) / 2
        })

### 📈 Extracted Data Insights
The pipeline succe![Sentiment Analysis Bar Chart](sentiment_chart.png)ssfully processed the raw footage and extracted **103 unique entities**, utilizing tracking IDs to completely eliminate multi-frame duplication errors. 

## Visualisation: **Traffic Volume Breakdown:**
![Chart](trffic_plot.png)


### 💡 Key Insights & Business Value
* **Unstructured to Structured:** This project proves the viability of turning completely unstructured data (video pixels) into highly structured tabular data (CSV) without human intervention.
* **High-Fidelity Tracking:** The pipeline successfully handled the "multi-frame duplication" problem natively. For example, a single car moving across 300 individual frames was correctly logged as exactly 1 unique vehicle in the final EDA.
* **Scalability & ROI:** This exact architecture can be deployed on live IP camera feeds (via RTSP) to create real-time, automated municipal dashboards for smart-city initiatives, turning hours of manual human counting into seconds of automated compute.