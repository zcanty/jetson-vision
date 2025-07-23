import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Images
imgs = ['/home/ecanty/Pictures/cars/2026-koenigsegg-sadairs-spear-4k-front-view-exterior-wallpapers4screen.com-1920x1080.jpg']  # batch of images

# Inference
results = model(imgs)

# Display results in pandas dataframe
print(results.pandas().xyxy)  