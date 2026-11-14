
import os
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

# Dataset paths
CSV_PATH = "dataset/IFND.csv"
OUTPUT_DIR = "dataset/images"

TARGET = 100
IMAGE_SIZE = (224, 224)

os.makedirs(f"{OUTPUT_DIR}/FAKE", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/REAL", exist_ok=True)

# Count existing images
def count_images(folder):
    return len([
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

fake_count = count_images(f"{OUTPUT_DIR}/FAKE")
real_count = count_images(f"{OUTPUT_DIR}/REAL")

print("Existing FAKE:", fake_count)
print("Existing REAL:", real_count)

# Stop if target already reached
if fake_count >= TARGET and real_count >= TARGET:
    print("Already reached 100 FAKE and 100 REAL!")
else:
    df = pd.read_csv(CSV_PATH, encoding="latin-1")

    df["Label"] = df["Label"].astype(str).str.strip().str.upper()
    df["Image"] = df["Image"].astype(str).str.strip()

    df = df[df["Label"].isin(["FAKE", "TRUE"])]
    df = df[df["Image"].str.startswith(("http://", "https://"))]
    df = df.drop_duplicates(subset=["Image"])

    # Select candidates
    fake = df[df["Label"] == "FAKE"].sample(
        frac=1, random_state=42
    )

    real = df[df["Label"] == "TRUE"].sample(
        frac=1, random_state=42
    )

    selected = [
        ("FAKE", fake),
        ("REAL", real)
    ]

    success = {"FAKE": 0, "REAL": 0}
    failed = 0

    for label, data in selected:
        folder = os.path.join(OUTPUT_DIR, label)
        current = count_images(folder)
        needed = max(0, TARGET - current)

        print(f"\nDownloading {needed} {label} images...")

        for _, row in tqdm(data.iterrows(), total=len(data)):
            if success[label] >= needed:
                break

            url = row["Image"]

            try:
                response = requests.get(
                    url,
                    timeout=12,
                    headers={"User-Agent": "Mozilla/5.0"}
                )
                response.raise_for_status()

                if len(response.content) > 5 * 1024 * 1024:
                    failed += 1
                    continue

                image = Image.open(
                    BytesIO(response.content)
                ).convert("RGB")

                image = image.resize(IMAGE_SIZE)

                # Next unused filename
                existing_numbers = []
                for name in os.listdir(folder):
                    base = os.path.splitext(name)[0]
                    if base.startswith(label.lower() + "_"):
                        try:
                            existing_numbers.append(
                                int(base.split("_")[-1])
                            )
                        except ValueError:
                            pass

                next_number = max(existing_numbers, default=0) + 1
                filename = f"{label.lower()}_{next_number:03d}.jpg"

                save_path = os.path.join(folder, filename)

                image.save(
                    save_path,
                    "JPEG",
                    quality=85,
                    optimize=True
                )

                success[label] += 1

            except Exception:
                failed += 1

    print("\nDOWNLOAD COMPLETED")
    print("New FAKE:", success["FAKE"])
    print("New REAL:", success["REAL"])
    print("Failed:", failed)
    print("Final FAKE:", count_images(f"{OUTPUT_DIR}/FAKE"))
    print("Final REAL:", count_images(f"{OUTPUT_DIR}/REAL"))
