# import tensorflow as tf
# from keras.applications import ResNet50
# from keras.layers import Dense, GlobalAveragePooling2D, Dropout
# from keras.models import Model

# CATEGORIES = ["topwear", "bottomwear", "shoes", "dress", "accessories", "outerwear"]
# NUM_CLASSES = len(CATEGORIES)

# def build_resnet_classifier():
#     base = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

#     # freeze base layers
#     for layer in base.layers:
#         layer.trainable = False

#     x = base.output
#     x = GlobalAveragePooling2D()(x)
#     x = Dropout(0.3)(x)
#     output = Dense(NUM_CLASSES, activation="softmax")(x)

#     model = Model(inputs=base.input, outputs=output)
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(1e-4),
#         loss="categorical_crossentropy",
#         metrics=["accuracy"]
#     )
#     return model

# from keras.applications.resnet50 import preprocess_input
# from keras.preprocessing.image import load_img, img_to_array
# import numpy as np

# def preprocess_image(img_path):
#     img = load_img(img_path, target_size=(224, 224))
#     img_array = img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = preprocess_input(img_array)
#     return img_array

# def predict_category(img_path, model):
#     img = preprocess_image(img_path)
#     pred = model.predict(img, verbose=0)[0]
#     class_idx = np.argmax(pred)
#     return CATEGORIES[class_idx]

# from keras.preprocessing.image import ImageDataGenerator

# def train_resnet(model, train_dir, val_dir, batch_size=16, epochs=10):
#     datagen = ImageDataGenerator(
#         rescale=1./255,
#         horizontal_flip=True,
#         rotation_range=15,
#         width_shift_range=0.1,
#         height_shift_range=0.1,
#     )

#     train_gen = datagen.flow_from_directory(
#         train_dir,
#         target_size=(224, 224),
#         batch_size=batch_size,
#         class_mode='categorical',
#         classes=CATEGORIES
#     )

#     val_gen = datagen.flow_from_directory(
#         val_dir,
#         target_size=(224, 224),
#         batch_size=batch_size,
#         class_mode='categorical',
#         classes=CATEGORIES
#     )

#     model.fit(
#         train_gen,
#         validation_data=val_gen,
#         epochs=epochs
#     )

#     return model


# # load
# from keras.models import load_model
# category_model = load_model("resnet_category_model.h5")

# # save
# category_model.save("resnet_category_model.h5")

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

