# app.py
from flask import Flask, request, jsonify
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
import io
from PIL import Image

app = Flask(__name__)

# Load pretrained model once
model = MobileNetV2(weights="imagenet")

# Known cat and dog labels from ImageNet
CAT_LABELS = {"tabby", "tiger_cat", "Persian_cat", "Siamese_cat", "Egyptian_cat"}
DOG_LABELS = {l for l in [
    "Chihuahua", "Japanese_spaniel", "Maltese_dog", "Pekinese", "Shih-Tzu",
    "Blenheim_spaniel", "papillon", "toy_terrier", "Rhodesian_ridgeback",
    "Afghan_hound", "basset", "beagle", "bloodhound", "bluetick",
    "black-and-tan_coonhound", "Walker_hound", "English_foxhound",
    "redbone", "borzoi", "Irish_wolfhound", "Italian_greyhound",
    "whippet", "Ibizan_hound", "Norwegian_elkhound", "otterhound",
    "Saluki", "Scottish_deerhound", "Weimaraner", "Staffordshire_bullterrier",
    "American_Staffordshire_terrier", "Bedlington_terrier", "Border_terrier",
    "Kerry_blue_terrier", "Irish_terrier", "Norfolk_terrier", "Norwich_terrier",
    "Yorkshire_terrier", "wire-haired_fox_terrier", "Lakeland_terrier",
    "Sealyham_terrier", "Airedale", "cairn", "Australian_terrier",
    "Dandie_Dinmont", "Boston_bull", "miniature_schnauzer", "giant_schnauzer",
    "standard_schnauzer", "Scotch_terrier", "Tibetan_terrier",
    "silky_terrier", "soft-coated_wheaten_terrier", "West_Highland_white_terrier",
    "Lhasa", "flat-coated_retriever", "curly-coated_retriever",
    "golden_retriever", "Labrador_retriever", "Chesapeake_Bay_retriever",
    "German_short-haired_pointer", "vizsla", "English_setter", "Irish_setter",
    "Gordon_setter", "Brittany_spaniel", "clumber", "English_springer",
    "Welsh_springer_spaniel", "cocker_spaniel", "Sussex_spaniel",
    "Irish_water_spaniel", "kuvasz", "schipperke", "groenendael",
    "malinois", "briard", "kelpie", "komondor", "Old_English_sheepdog",
    "Shetland_sheepdog", "collie", "Border_collie", "Bouvier_des_Flandres",
    "Rottweiler", "German_shepherd", "Doberman", "miniature_pinscher",
    "Greater_Swiss_Mountain_dog", "Bernese_mountain_dog", "Appenzeller",
    "EntleBucher", "boxer", "bull_mastiff", "Tibetan_mastiff", "French_bulldog",
    "Great_Dane", "Saint_Bernard", "Eskimo_dog", "malamute", "Siberian_husky",
    "dalmatian", "affenpinscher", "monkey", "pug", "Leonberg", "Newfoundland",
    "Great_Pyrenees", "Samoyed", "Pomeranian", "chow", "keeshond",
    "Brabancon_griffon", "Pembroke", "Cardigan", "toy_poodle",
    "miniature_poodle", "standard_poodle"
]}

def classify_image(img: Image.Image) -> str:
    # Resize for MobileNetV2
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    decoded = decode_predictions(preds, top=3)[0]

    # Check predictions
    for _, label, prob in decoded:
        if label in CAT_LABELS:
            return "Cat"
        if label in DOG_LABELS:
            return "Dog"
    return label

@app.route("/classify", methods=["POST"])
def classify():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img = Image.open(io.BytesIO(file.read()))
    result = classify_image(img)
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)
