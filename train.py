import argparse
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from alzheimers_mri.constants import CLASS_NAMES, IMAGE_SIZE, MODEL_NAMES
from alzheimers_mri.evaluation import calculate_metrics, save_metrics
from alzheimers_mri.models import build_model


def canonicalize_imagefolder(dataset):
    """Make ImageFolder targets use the app's documented class order."""
    if set(dataset.classes) != set(CLASS_NAMES):
        raise ValueError(
            "Each split must contain exactly these class folders: "
            + ", ".join(CLASS_NAMES)
        )
    missing = [
        name
        for name in CLASS_NAMES
        if not any(target == dataset.class_to_idx[name] for target in dataset.targets)
    ]
    if missing:
        raise ValueError("Each class folder needs at least one image; empty: " + ", ".join(missing))
    old_to_new = {
        dataset.class_to_idx[name]: index for index, name in enumerate(CLASS_NAMES)
    }
    dataset.samples = [
        (path, old_to_new[target]) for path, target in dataset.samples
    ]
    dataset.imgs = dataset.samples
    dataset.targets = [target for _, target in dataset.samples]
    dataset.classes = list(CLASS_NAMES)
    dataset.class_to_idx = {name: index for index, name in enumerate(CLASS_NAMES)}


class HuggingFaceImageDataset(Dataset):
    def __init__(self, dataset, transform):
        names = dataset.features["label"].names
        expected_names = {
            "mild_demented": "mild",
            "moderate_demented": "moderate",
            "non_demented": "non_demented",
            "very_mild_demented": "very_mild",
        }
        source_to_class = {}
        for label_index, label_name in enumerate(names):
            canonical_name = expected_names.get(label_name.lower())
            if canonical_name is None:
                raise ValueError(f"Unsupported Hugging Face class label: {label_name}")
            source_to_class[label_index] = CLASS_NAMES.index(canonical_name)
        if set(source_to_class.values()) != set(range(len(CLASS_NAMES))):
            raise ValueError("Hugging Face dataset must contain all four dementia classes.")
        self.dataset = dataset
        self.transform = transform
        self.source_to_class = source_to_class
        self.class_to_idx = {
            class_name: index for index, class_name in enumerate(CLASS_NAMES)
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset[index]
        image = row["image"].convert("RGB")
        return self.transform(image), self.source_to_class[int(row["label"])]


def main():
    parser = argparse.ArgumentParser(
        description="Train MRI classifiers from Hugging Face or local ImageFolder data."
    )
    parser.add_argument(
        "--source",
        choices=("huggingface", "local"),
        default="huggingface",
        help="Dataset source (default: Hugging Face Alzheimer MRI dataset)",
    )
    parser.add_argument("--data", default="data", help="Local ImageFolder dataset root")
    parser.add_argument(
        "--hf-dataset",
        default="Falah/Alzheimer_MRI",
        help="Hugging Face dataset ID (four-class brain MRI labels required)",
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", default="models")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("--epochs, --batch-size, and --learning-rate must be positive.")

    normalization = transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
            transforms.ToTensor(),
            normalization,
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            normalization,
        ]
    )
    test_data = None
    if args.source == "huggingface":
        from datasets import load_dataset

        print(f"Loading {args.hf_dataset}; first run downloads and caches the dataset.")
        splits = load_dataset(args.hf_dataset)
        if "train" not in splits or "test" not in splits:
            parser.error("Hugging Face dataset must provide train and test splits.")
        try:
            source_labels = splits["train"]["label"]
            train_indices, val_indices = train_test_split(
                range(len(splits["train"])),
                test_size=0.1,
                random_state=42,
                stratify=source_labels,
            )
            train_split = splits["train"].select(train_indices)
            val_split = splits["train"].select(val_indices)
            train_data = HuggingFaceImageDataset(train_split, train_transform)
            val_data = HuggingFaceImageDataset(
                val_split,
                eval_transform,
            )
            test_data = HuggingFaceImageDataset(splits["test"], eval_transform)
        except (ValueError, KeyError) as error:
            parser.error(f"Could not use Hugging Face dataset: {error}")
    else:
        root = Path(args.data)
        train_path, val_path = root / "train", root / "val"
        if not train_path.is_dir() or not val_path.is_dir():
            parser.error("Expected --data/train and --data/val class-folder directories.")
        train_data = datasets.ImageFolder(train_path, transform=train_transform)
        val_data = datasets.ImageFolder(val_path, transform=eval_transform)
        try:
            canonicalize_imagefolder(train_data)
            canonicalize_imagefolder(val_data)
        except ValueError as error:
            parser.error(str(error))
        local_test_path = root / "test"
        if local_test_path.is_dir():
            test_data = datasets.ImageFolder(local_test_path, transform=eval_transform)
            canonicalize_imagefolder(test_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    for model_name in args.models:
        train_loader = DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
        )
        val_loader = DataLoader(
            val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
        )
        model = build_model(model_name, pretrained=True).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        criterion = nn.CrossEntropyLoss()
        best_accuracy = -1.0
        checkpoint_path = output_dir / f"{model_name}.pth"

        for epoch in range(args.epochs):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()

            model.eval()
            targets, predictions = [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    logits = model(images.to(device))
                    predictions.extend(logits.argmax(dim=1).cpu().tolist())
                    targets.extend(labels.tolist())
            accuracy = float(accuracy_score(targets, predictions))
            print(f"{model_name} epoch {epoch + 1}/{args.epochs}: val accuracy {accuracy:.4f}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(
                    {
                        "model_name": model_name,
                        "state_dict": model.state_dict(),
                        "class_to_idx": train_data.class_to_idx,
                        "image_size": IMAGE_SIZE,
                        "validation_accuracy": best_accuracy,
                    },
                    checkpoint_path,
                )

        if test_data is not None:
            test_loader = DataLoader(
                test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
            )
            model.load_state_dict(torch.load(checkpoint_path, map_location=device)["state_dict"])
            model.eval()
            targets, predictions = [], []
            with torch.no_grad():
                for images, labels in test_loader:
                    predictions.extend(model(images.to(device)).argmax(dim=1).cpu().tolist())
                    targets.extend(labels.tolist())
            metrics = calculate_metrics(targets, predictions)
            metrics["model_name"] = model_name
            metrics["evaluation_split"] = "test"
            metrics["dataset"] = args.hf_dataset if args.source == "huggingface" else str(root)
            save_metrics(metrics, output_dir / f"{model_name}_metrics.json")
            print(f"Saved held-out test metrics to {output_dir / f'{model_name}_metrics.json'}")
        else:
            print("No test split provided; held-out metrics were not generated.")


if __name__ == "__main__":
    main()
