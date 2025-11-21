import numpy as np
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from keras.preprocessing import image

# Load once
model = MobileNetV2(weights="imagenet")

# We only keep 2 classes
TARGET_CATEGORIES = {
    "topwear": [
        "tshirt", "jersey", "sweatshirt", "shirt", "cardigan", "top", "trench_coat", "coat"
    ],
    "bottomwear": [
        "jean", "jeans", "miniskirt", "skirt", "pants", "trousers", "shorts"
    ],
}

def normalize(x):
    return str(x).lower().replace(" ", "_").replace("-", "_")

def predict_category(img_path):
    """
    Returns only: "topwear", "bottomwear", or "unknown"
    """

    try:
        img = image.load_img(img_path, target_size=(224, 224))
        arr = image.img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)
    except:
        return "unknown"

    preds = model.predict(arr, verbose=0)
    decoded = decode_predictions(preds, top=5)[0]

    for (_, name, score) in decoded:
        name = normalize(name)

        # check topwear
        for tw in TARGET_CATEGORIES["topwear"]:
            if tw in name:
                return "topwear"

        # check bottomwear
        for bw in TARGET_CATEGORIES["bottomwear"]:
            if bw in name:
                return "bottomwear"

    return "unknown"
