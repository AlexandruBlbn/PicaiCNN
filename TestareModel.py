import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
import numpy as np
from UNet import UNet
from metrici import mean_dice_score

def color_to_class_mapping(arr):
    """
    Convertește un array RGB într-o hartă de etichete bazată pe culori predefinite.
    """
    color2class = {
        (0, 0, 0): 0,         # Fundal
        (127, 127, 127): 1,   # Zonă periferică
        (255, 255, 255): 2    # Prostată
    }

    h, w, _ = arr.shape
    label_np = np.zeros((h, w), dtype=np.int64)
    for i in range(h):
        for j in range(w):
            pixel = tuple(arr[i, j])
            label_np[i, j] = color2class.get(pixel, 0)
    return label_np

def test_single_image(model_path, image_path, label_path=None, device="cpu"):

    # Încarcă modelul
    model = UNet(n_channels=3, n_seg_classes=3, n_bin_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()


    transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)  

    if label_path:
        label = Image.open(label_path).convert("RGB")
        label_transform = T.Compose([
            T.Resize((128, 128), interpolation=T.InterpolationMode.NEAREST)
        ])
        label_np = np.array(label_transform(label))
        label_tensor = color_to_class_mapping(label_np)
    else:
        label_tensor = None

    # Obține predicția modelului
    with torch.no_grad():
        seg_out, bin_out = model(input_tensor)
        preds = torch.argmax(seg_out, dim=1).squeeze(0).cpu().numpy()
        gleason_score = int(torch.sigmoid(bin_out).item() > 0.5)  # Scor Gleason (0 sau 1)

    if label_tensor is not None:
        if preds.shape != label_tensor.shape:
            raise ValueError(f"Dimensiunile predicției ({preds.shape}) și ale ground truth-ului ({label_tensor.shape}) nu se potrivesc.")

        dice_score = mean_dice_score(torch.tensor(preds).unsqueeze(0),
                                     torch.tensor(label_tensor).unsqueeze(0).long(),
                                     num_classes=3).item()
        print(f"Dice Score: {dice_score:.4f}")
    else:
        dice_score = None

    print(f"Scor Gleason (predicție): {gleason_score}")
    fig, axes = plt.subplots(1, 3 if label_tensor is not None else 2, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Imaginea de intrare")
    axes[0].axis("off")

    if label_tensor is not None:
        axes[1].imshow(label_tensor, cmap="gray")
        axes[1].set_title("Segmentare reală")
        axes[1].axis("off")

    pred_ax = axes[2] if label_tensor is not None else axes[1]
    pred_ax.imshow(preds, cmap="gray")
    pred_title = f"Predictie: Model | Gleason: {gleason_score}"
    if label_tensor is not None:
        pred_title += f" | Dice: {dice_score:.4f}"
    pred_ax.set_title(pred_title)
    pred_ax.axis("off")

    plt.tight_layout()
    plt.show()

test_single_image(
    model_path="results/ponderi_model_fold_3.pth",
    image_path=r"F:\PICAT\ProiectPython\data\data\10000_1000000_t2w.png",
    label_path=r"F:\PICAT\ProiectPython\data\label\10000_1000000.png"
)
