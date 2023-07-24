from ultralytics import YOLO
from datasets import load_dataset
from PIL import Image

dataset = load_dataset("Kili/plastic_in_river")
img = dataset['test'][0]['image']

model = YOLO(r"runs\detect\train\weights\best.pt")

res = model.predict(img)[0]

print(res.boxes)

res = res.plot(line_width=1)
res= res[:,:,::-1]
res = Image.fromarray(res)
res.save("outputt.png")