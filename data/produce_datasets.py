from sklearn.model_selection import train_test_split
import json

with open('annotations/cropped_persons_annotations.json', mode='r') as f:
    annotations = json.load(f)

print(f"Total items: {len(annotations)}")

# First: split off test (15%)
train_val, test_data = train_test_split(
    annotations, test_size=0.15, random_state=42
)

# Now: split train vs val (15% of original)
val_size = 0.15 / (1 - 0.15)  # scale 15% relative to remaining 85%
train_data, val_data = train_test_split(
    train_val, test_size=val_size, random_state=42
)

print(f"Train: {len(train_data)}")
print(f"Val:   {len(val_data)}")
print(f"Test:  {len(test_data)}")

with open('annotations/train_annotations_full.json', mode='w+') as f:
    json.dump(train_data, f)

with open('annotations/test_annotations_full.json', mode='w+') as f:
    json.dump(test_data, f)

with open('annotations/val_annotations_full.json', mode='w+') as f:
    json.dump(val_data, f)