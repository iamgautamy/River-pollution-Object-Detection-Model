#different bounding box formats

#x_min, y_min , x_max , y_max
#x1 , y1 , x2 , y2
#x , y , w , h

#yolov8 format (n-> normalized) 
# x_center_n , y_center_n , bbox_w_n , bbox_h_n

#conversion of x , y , w , h to yolo format is

# (x + bbox_w /2) / w (w is the image width) 
# (y + bbox_h/2) / h (h is the height of the image)
# bbox_w / w
# bbox_h / h 
import os
from datasets import load_dataset

def dumb_images_and_labels(data,split):
    data = data[split]
    for i,example in enumerate(data):
        image = example['image']
        labels = example['litter']['label']
        bboxes = example['litter']['bbox']
        targets = []
        for label,box in zip(labels,bboxes):
            targets.append(f"{label} {box[0]} {box[1]} {box[2]} {box[3]}")
        with open(f"datasets/labels/{split}/{i}.txt","w",) as f:
            for target in targets:
                f.write(target +  "\n")
        image.save(f"datasets/images/{split}/{i}.png")     

dataset = load_dataset("Kili/plastic_in_river")
print("dataset loaded")
os.makedirs("datasets/images/train",exist_ok=True)
os.makedirs("datasets/images/validation",exist_ok=True)
os.makedirs("datasets/labels/train",exist_ok=True)
os.makedirs("datasets/labels/validation",exist_ok=True)

dumb_images_and_labels(dataset,"train")
dumb_images_and_labels(dataset,"validation")