from sklearn.model_selection import train_test_split
import json

with open('annotations/cropped_persons_annotations.json', mode='r') as f:
    annotations = json.load(f)

annotations = annotations[:int(len(annotations) * 0.1)]

print(f"Total items: {len(annotations)}")

train_data, test_data = train_test_split(annotations, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.05, random_state=42)

print(f"Train data items: {len(train_data)}")
print(f"Val data items: {len(val_data)}")
print(f"Test data items: {len(test_data)}")

with open('annotations/train_annotations.json', mode='w+') as f:
    json.dump(train_data, f)

with open('annotations/test_annotations.json', mode='w+') as f:
    json.dump(test_data, f)

with open('annotations/val_annotations.json', mode='w+') as f:
    json.dump(val_data, f)