
import os
from PIL import Image

BASE_DIR = "dataset/images"

for label in ["FAKE", "REAL"]:
    folder = os.path.join(BASE_DIR, label)
    valid = 0
    invalid = 0

    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)

        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        try:
            with Image.open(path) as img:
                img.verify()
            valid += 1
        except Exception:
            invalid += 1

    print(f"\n{label}:")
    print("Valid:", valid)
    print("Invalid:", invalid)

print("\nVerification completed!")
