import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import os


# create the model using the saved model weights and load it into memory
def load_model(model_path, device, num_labels):
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # put the model into evaluation mode
    return model


# define the function for image classification
def classify_image(image_path, model, feature_extractor, device, class_names):
    # open the image and convert it to RGB format
    image = Image.open(image_path).convert("RGB")

    # resize the image if it is not in size (224, 224)
    if image.size != (224, 224):
        image = image.resize((224, 224))

    # feature extractor is used to convert the image to the format expected by the model
    inputs = feature_extractor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    # run the model in evaluation mode and get the prediction
    with torch.no_grad():
        outputs = model(pixel_values)

    # make an estimation by choosing the highest logit value
    predicted_idx = outputs.logits.argmax(dim=-1).item()
    predicted_class = class_names[predicted_idx]

    return predicted_class


if __name__ == "__main__":
    # device setting: use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # the path to the trained model weight file
    model_path = "vit_ucmerced.pth"

    # Number of classes determined during training
    num_labels = 21

    class_names = ["agricultural", "airplane", "baseballdiamond", "beach", "buildings",
                   "chaparral", "denseresidential", "forest", "freeway", "golfcourse",
                   "harbor", "intersection", "mediumresidential", "mobilehomepark",
                   "overpass", "parkinglot", "river", "runway", "sparseresidential",
                   "storagetanks", "tenniscourt"]

    # load the model
    model = load_model(model_path, device, num_labels)

    # reload the feature extractor used in the training
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

    # process all ".png" files in the "screenshots" folder
    screenshots_dir = "screenshots"
    for filename in os.listdir(screenshots_dir):
        if filename.endswith(".png") or filename.endswith(".PNG"):
            image_path = os.path.join(screenshots_dir, filename)
            predicted_label = classify_image(image_path, model, feature_extractor, device, class_names)
            print(f"{filename} için tahmin edilen sınıf: {predicted_label}")

    print("Sınıflandırma tamamlandı.")
