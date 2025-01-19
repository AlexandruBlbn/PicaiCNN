import torch



def dice_score_per_class(pred, target, num_classes=3):
    """
    Calculates the Dice score for each class.
    pred and target should be tensors of shape (N, H, W) or (H, W),
    with integer class labels from 0..(num_classes-1).
    """
    # Flatten the tensors to make calculation easier
    pred = pred.view(-1)
    target = target.view(-1)
    dice_scores = []

    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        intersection = (pred_c & target_c).sum().float()
        union = pred_c.sum().float() + target_c.sum().float()
        dice = (2.0 * intersection) / (union + 1e-7) if union > 0 else 1.0
        dice_scores.append(dice)

    return dice_scores

def mean_dice_score(pred, target, num_classes=4):
    """
    Calculates the mean Dice score across all classes.
    """
    scores = dice_score_per_class(pred, target, num_classes)
    return sum(scores) / len(scores)

def sensitivity_per_class(pred, target, num_classes=4):
    """
    Calculates sensitivity (TP / (TP + FN)) for each class.
    """
    pred = pred.view(-1)
    target = target.view(-1)
    sens_scores = []

    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        tp = (pred_c & target_c).sum().float()
        fn = (~pred_c & target_c).sum().float()
        sens = tp / (tp + fn + 1e-7) if (tp + fn) > 0 else 0.0
        sens_scores.append(sens.item())

    return sens_scores

def accuracy_score(pred, target):
    """
    Calculates the accuracy across all classes (correct predictions / total pixels).
    """
    correct = (pred == target).sum().float()
    total = pred.numel()
    return correct / (total + 1e-7)

def compute_metrics(pred, target, num_classes=4):
    """
    Consolidates multiple metrics (accuracy, sensitivity, Dice per class, mean Dice)
    into a single dictionary for convenience.
    """
    # pred and target should have the same shape: (N, H, W) or (H, W)
    # with values in 0..(num_classes-1)
    dice_per_cls = dice_score_per_class(pred, target, num_classes)
    mean_dice = mean_dice_score(pred, target, num_classes)
    sensitivity_list = sensitivity_per_class(pred, target, num_classes)
    acc = accuracy_score(pred, target)

    # Specificity is optional. If you need it, you can define it similarly
    # to how sensitivity is computed (TN / (TN + FP)).

    return {
        "accuracy": acc.item(),
        "sensitivity": sensitivity_list,
        "dice_per_class": dice_per_cls,
        "mean_dice": mean_dice
    }