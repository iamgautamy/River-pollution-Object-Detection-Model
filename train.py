from ultralytics import YOLO
if __name__ == '__main__':
    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

    # Train the model
    model.train(data="plastic.yaml", epochs=20, batch=2 ,imgsz=(1280,720), workers=8, optimizer='Adam' ,pretrained=True, val=True, plots=True, save=True, show=True, optimize=True, lr0=0.001 , amp=False)
 