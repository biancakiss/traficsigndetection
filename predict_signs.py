"""
Traffic Sign Prediction
========================
Pune imaginile tale intr-un folder numit 'test_images'
apoi ruleaza scriptul si vei vedea ce semn ghiceste modelul.

Structura asteptata:
    traffic_sign_model.keras   <- modelul antrenat
    test_images/
        poza1.jpg
        poza2.jpg
        ...
    predict_signs.py           <- acest fisier
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from datasets import load_dataset

# ──────────────────────────────────────────
# SETARI
# ──────────────────────────────────────────
MODEL_PATH  = "traffic_sign_model.keras"
TEST_FOLDER = "test_images"
IMAGE_SIZE  = (224, 224)


CLASS_NAMES = [
    'animals',                              # 0
    'construction',                         # 1
    'cycles crossing',                      # 2
    'danger',                               # 3
    'no entry',                             # 4
    'pedestrian crossing',                  # 5
    'school crossing',                      # 6
    'snow',                                 # 7
    'stop',                                 # 8
    'bend',                                 # 9
    'bend left',                            # 10
    'bend right',                           # 11
    'give way',                             # 12
    'go left',                              # 13
    'go left or straight',                  # 14
    'go right',                             # 15
    'go right or straight',                 # 16
    'go straight',                          # 17
    'keep left',                            # 18
    'keep right',                           # 19
    'no overtaking',                        # 20
    'no overtaking -trucks-',               # 21
    'no traffic both ways',                 # 22
    'no trucks',                            # 23
    'priority at next intersection',        # 24
    'priority road',                        # 25
    'restriction ends',                     # 26
    'restriction ends -overtaking -trucks--', # 27
    'restriction ends -overtaking-',        # 28
    'restriction ends 80',                  # 29
    'road narrows',                         # 30
    'roundabout',                           # 31
    'slippery road',                        # 32
    'speed limit 100',                      # 33
    'speed limit 120',                      # 34
    'speed limit 20',                       # 35
    'speed limit 30',                       # 36
    'speed limit 50',                       # 37
    'speed limit 60',                       # 38
    'speed limit 70',                       # 39
    'speed limit 80',                       # 40
    'traffic signal',                       # 41
    'uneven road',                          # 42
]

# ──────────────────────────────────────────
# 1. INCARCA MODELUL
# ──────────────────────────────────────────
print("Se incarca modelul...")
if not os.path.exists(MODEL_PATH):
    print(f"EROARE: Nu gasesc modelul la '{MODEL_PATH}'")
    print("Asigura-te ca ai rulat training-ul si ca fisierul .keras e in acelasi folder.")
    exit(1)

model = tf.keras.models.load_model(MODEL_PATH)
print(f"Model incarcat! ({model.output_shape[-1]} clase)")

# ──────────────────────────────────────────
# 2. INCARCA NUMELE CLASELOR
# ──────────────────────────────────────────
print("Se incarca numele claselor...")
try:

    class_names = CLASS_NAMES
    print(f"{len(class_names)} clase incarcate.")
except Exception as e:
    print(f"Nu s-au putut incarca clasele: {e}")
    class_names = [f"Clasa {i}" for i in range(model.output_shape[-1])]

# ──────────────────────────────────────────
# 3. INCARCA IMAGINILE DIN FOLDER
# ──────────────────────────────────────────
if not os.path.exists(TEST_FOLDER):
    os.makedirs(TEST_FOLDER)
    print(f"\nFolderul '{TEST_FOLDER}' a fost creat.")
    print(f"Pune imaginile tale acolo si ruleaza din nou scriptul.")
    exit(0)

valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
files = sorted([
    f for f in os.listdir(TEST_FOLDER)
    if f.lower().endswith(valid_ext)
])

if len(files) == 0:
    print(f"\nNu am gasit nicio imagine in '{TEST_FOLDER}'.")
    print("Pune fisiere .jpg sau .png acolo si incearca din nou.")
    exit(0)

print(f"\nAm gasit {len(files)} imagini. Se proceseaza...\n")

# ──────────────────────────────────────────
# 4. PREDICTIE PENTRU FIECARE IMAGINE
# ──────────────────────────────────────────
results = []

for filename in files:
    path = os.path.join(TEST_FOLDER, filename)

    # citeste si pregateste imaginea
    img_bgr    = cv2.imread(path)
    img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resize = cv2.resize(img_rgb, IMAGE_SIZE)
    img_input  = np.expand_dims(img_resize.astype(np.float32) / 255.0, axis=0)

    # predictie
    probs    = model.predict(img_input, verbose=0)[0]
    top3_idx = np.argsort(probs)[::-1][:3]

    pred_name = class_names[top3_idx[0]]
    pred_conf = probs[top3_idx[0]] * 100

    print(f"  {filename}")
    print(f"    #1 -> {pred_name}  ({pred_conf:.1f}%)")
    print(f"    #2 -> {class_names[top3_idx[1]]} ({probs[top3_idx[1]]*100:.1f}%)")
    print(f"    #3 -> {class_names[top3_idx[2]]} ({probs[top3_idx[2]]*100:.1f}%)")
    print()

    results.append({
        "filename": filename,
        "image":    img_rgb,
        "top3_idx": top3_idx,
        "probs":    probs,
    })

# ──────────────────────────────────────────
# 5. AFISEAZA TOATE REZULTATELE VIZUAL
# ──────────────────────────────────────────
n     = len(results)
ncols = 2          # imagine  |  bar chart
fig, axes = plt.subplots(n, ncols, figsize=(12, n * 4))
fig.patch.set_facecolor("#111111")

# normalizare pentru o singura imagine
if n == 1:
    axes = [axes]

for row, res in enumerate(results):
    ax_img = axes[row][0]
    ax_bar = axes[row][1]

    # ── imaginea originala ────────────────
    ax_img.imshow(res["image"])
    ax_img.set_title(res["filename"], color="white", fontsize=10, pad=6)
    ax_img.axis("off")

    pred_label = class_names[res["top3_idx"][0]]
    conf       = res["probs"][res["top3_idx"][0]] * 100
    ax_img.text(
        0.5, -0.05,
        f"{pred_label}  —  {conf:.1f}%",
        transform=ax_img.transAxes,
        ha="center", color="#00e5a0",
        fontsize=11, fontweight="bold"
    )

    # ── bar chart top 3 ───────────────────
    ax_bar.set_facecolor("#1c1c2e")
    labels = [class_names[i][:25] for i in res["top3_idx"]]
    values = [res["probs"][i] * 100 for i in res["top3_idx"]]
    colors = ["#00e5a0", "#4a9eff", "#ff6b6b"]

    ax_bar.barh(labels[::-1], values[::-1], color=colors[::-1],
                height=0.5, edgecolor="none")

    for i, (label, val) in enumerate(zip(labels[::-1], values[::-1])):
        ax_bar.text(
            val + 0.5, i,
            f"{val:.1f}%", va="center", color="white", fontsize=9
        )

    ax_bar.set_xlim(0, 115)
    ax_bar.set_xlabel("Incredere (%)", color="#aaaaaa", fontsize=8)
    ax_bar.tick_params(axis="x", colors="#aaaaaa")
    ax_bar.tick_params(axis="y", colors="white", labelsize=8)
    for spine in ax_bar.spines.values():
        spine.set_visible(False)
    ax_bar.set_title("Top 3 predictii", color="white", fontsize=9, pad=6)

plt.suptitle("Traffic Sign Recognition", color="white",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("rezultate.png", dpi=150, bbox_inches="tight",
            facecolor="#111111")
plt.show()
print("Rezultatele au fost salvate in 'rezultate.png'")