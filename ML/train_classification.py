from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")

model.train(
    data="data_classification",
    epochs=50,
    imgsz=224,
    batch=32,
    device="cpu",      # change to 0 if using NVIDIA GPU
    name="exam_behaviour_cls"
)