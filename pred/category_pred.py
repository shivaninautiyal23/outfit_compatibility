from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np

model = MobileNetV2(weights="imagenet")

FASHION_MAP = {
    "jersey": "tshirt",
    "tshirt": "tshirt",
    "running_shoe": "shoes",
    "sandal": "shoes",
    "jean": "jeans",
    "trench_coat": "coat",
    "gown": "dress",
    "miniskirt": "skirt",
    "suit": "formal",
    "sunglass": "accessory",
    "shades": "accessory",
}

def predict_category(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    decoded = decode_predictions(preds, top=5)[0]

    for (_id, name, score) in decoded:
        if name in FASHION_MAP:
            return FASHION_MAP[name], float(score)

    # fallback
    return decoded[0][1], float(decoded[0][2])


