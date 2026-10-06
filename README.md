# outfit_compatibility

My project **Outfit Compatibility Checker** uses deep learning to understand clothing images and suggest if outfit is compatibile and also gives visually similar items to users. It analyzes product features using ResNet and MobileNet trained model and generates embeddings that help match styles, colours, and pattern more accurately.

# Industry revelance 
It can be used in E-commerce to boost sales, personalize the shopping experience.
It can be scaled to handle millions of products, power virtual try-ons and even integrate into AR/VR shopping apps.

# Tech Stack Used 
1. Keras
2. Tensorflow
3. ResNet
4. MobileNet
5. Sklearn
6. numpy
7. pandas
   
# About all the files 
1. **Dataset** -
   a. fashion_products_sample.csv - file containing custom dataset (obviously made by me and my team) having all the information about the clothes and accessories (like product_id, product_name, category, brand, gender, image_path, occassions,class, colour, description)
   b. pairing.csv - having two images img_a and img_b with label having value 0/1 where 0 means these two images are not compatible and 1 means the opposite.

2. **app.py** -
   This Flask file basically manages image uploads, routes them to ML functions, cleans up temporary data, and renders results for compatibility and recommendations.
   
3. **rec.py** -
   This file loads your trained compatibility model, manages product data, extracts ResNet50 embeddings, predicts fashion categories, calculates outfit compatibility, and returns top recommendations using cached features for faster performance.
   
4. **preprocess.py** -
   This file basically have a function that is being used in train.py for extracting features of the images.
   This function basically resize images into 224 X 224 (ResNet rule) , then normalize pixel values [0-1] , sends it through ResNet50 (Residual Network of 50 layers) , gets 2048 dimensional embedding and return that vector.

5. **train.py** -
   Training an outfit compatibility model.
   Takes two clothing images -> extract features -> learns if the pair is compatible or not.
   a. loads data [ both csv files and images]
   b. extract image features (use fn - get_feature_database)
   c. prepares training pairs ( fn - get_pair_features, uses pairing.csv, for every pair (img_a, img_b) combines and get 4096 features)
   d. clean data by skipping missing image and those whose features can't be extracted.
   e. split data [ 60% training, 20% validation, 20% testing]
   f. defines the neural network { 4096 inputs -> dense : 512 neurons + ReLU (rectified linear unit) -> dropout -> dense: 128 neurons + ReLU -> dense: 1 neurons + sigmoid }
   g. trains the model (50 epochs) try to reduce binary crossentropy loss and increase the accuracy
   h. evaluates on test model - accuracy = 82.42%
   i. saves the model(model.h5) 

**FLOW:**
Garment Images
↓
ResNet50
↓
2048-dimensional image embedding for each garment
↓
Garment A embedding + Garment B embedding
↓
4096-dimensional pair vector
↓
Dense Neural Network
↓
Sigmoid
↓
Compatibility Score / Compatible (1) or Not Compatible (0)
6. **pred/cat.py** -
   This file is a clothing classifier that uses MobileNetV2’s ImageNet labels to decide whether your input image is topwear, bottomwear, or unknown, by simply matching predicted labels with a manual list.

