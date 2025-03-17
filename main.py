if __name__ == '__main__':
    import torch
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, random_split

    # pre-processing that will be used for Vision Transformer (resize, tensor and normalize)
    from transformers import ViTFeatureExtractor

    # load the pre-processor of the ViT model
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

    # resize the image to the size expected by the model (224x224) and normalize it
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    ])

    # directory where the dataset is located
    data_dir = "C:/Users/pc/Downloads/UCMerced/UCMerced_LandUse/Images"

    # ImageFolder automatically tags and uploads images in each folder
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    print("Toplam resim sayısı:", len(dataset))
    print("Sınıflar:", dataset.classes)

    # separate the training and test data sets by 80% - 20%
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # create a DataLoader, specify the batch size and shuffle settings
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

    # DEFINING THE VISION TRANSFORMER MODEL
    from transformers import ViTForImageClassification

    # get the number of classes in the dataset
    num_labels = len(dataset.classes)
    print("Sınıf sayısı:", num_labels)

    # load the pre-trained ViT model and update the num_labels parameter
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )

    # device setting: use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # TRAINING
    import torch.optim as optim

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 5  # defining the number of epochs

    model.train()  # put the model into training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in train_loader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, labels=labels)
            loss = outputs.loss  # the loss defined by the model itself is used
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # EVALUATION
    model.eval()  # put the model into evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            # select the class with the highest value from the logits in the output
            predictions = outputs.logits.argmax(dim=-1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print("Test Doğruluğu: {:.2f}%".format(accuracy))

    # save the model
    torch.save(model.state_dict(), "vit_ucmerced.pth")
