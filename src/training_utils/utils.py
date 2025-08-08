import os
# AI - libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
import numpy as np

from architecture.generator import attention_unet
from architecture.discriminator import Discriminator
from dataset.training_dataset import *
import mlflow



def initialize_model_and_weight(img_shape: tuple = (3, 128, 128), num_classes: int = 10, device: torch.device = 'cuda:0'):
    '''
    Initialize the model and weights for the generator and discriminator networks.
    :param img_shape: the shape of the input image (channels, height, width)
    :param res_blocks: the number of residual blocks in the discriminator network
    :param num_classes: the number of classes in the dataset
    :param device: the device to run the model on
    :return: dictionary containing the generator, discriminator and classifier models
    '''
    # Generator
    generator = attention_unet(num_classes=num_classes).to(device)

    # Discriminator
    discriminator = Discriminator(img_shape=img_shape, c_dim=num_classes).to(device)

    # Classifier
    classifier = efficientnet_b3(weights=EfficientNet_B3_Weights)
    classifier.classifier = nn.Sequential(nn.Dropout(p=0.3, inplace=True),
                                     nn.Linear(in_features=1536, out_features=num_classes))  # b3
    classifier = classifier.to(device)

    models = {
        'generator': generator,
        'discriminator': discriminator,
        'classifier': classifier
    }

    return models

def initialize_optimzers(models: dict, lr: float, betas: tuple = (0.9, 0.999)):
    '''
    Initialize the optimizers for the generator, discriminator and classifier networks.
    :param models: dict containing the generator, discriminator and classifier models
    :param lr: the learning rate for the optimizers
    :param betas: the betas for the AdamW optimizer
    :return: dict containing the generator, discriminator and classifier optimizers
    '''

    optimizer_G = torch.optim.AdamW(models['generator'].parameters(), lr=lr, betas=betas)
    optimizer_D = torch.optim.AdamW(models['discriminator'].parameters(), lr=lr, betas=betas)
    optimizer_C = torch.optim.AdamW(models['classifier'].parameters(), lr=lr, betas=betas, weight_decay=1e-4)

    optimizers = {
        'generator': optimizer_G,
        'discriminator': optimizer_D,
        'classifier': optimizer_C
    }

    return optimizers

def load_dataset(path_to_dataset: str, batch_size: int):
    '''
    Load the dataset from the given path.
    :param path_to_dataset: the path to the dataset
    :param batch_size: the batch size for the DataLoader
    :param supervised:
    :return: DataLoader object for the dataset
    '''

    supervised_training_folder = "training_supervised"
    unsupervised_training_folder = "training_unsupervised"

    supervised_train_path = os.path.join(path_to_dataset, supervised_training_folder)
    unsupervised_training_folder = os.path.join(path_to_dataset, unsupervised_training_folder)
    validation_path = os.path.join(path_to_dataset, 'validation')

    supervised_training_loader = get_loader(supervised_train_path, batch_size)
    unsupervised_training_loader = get_loader(unsupervised_training_folder, batch_size)
    validation_loader = get_loader(validation_path, batch_size)

    if 'bloodmnist' in path_to_dataset:
        n_channels = 3
        n_classes = 8
    if 'pathmnist' in path_to_dataset:
        n_channels = 3
        n_classes = 9
    if 'retinamnist' in path_to_dataset:
        n_channels = 3
        n_classes = 5
    if 'dermamnist' in path_to_dataset:
        n_channels = 3
        n_classes = 7
    if 'tissuemnist' in path_to_dataset:
        n_channels = 3
        n_classes = 8
    if 'breastmnist' in path_to_dataset:
        n_channels = 1
        n_classes = 2
    if 'chestmnist' in path_to_dataset:
        n_channels = 1
        n_classes = 15
    if 'octmnist' in path_to_dataset:
        n_channels = 1
        n_classes = 4
    if 'organamnist' in path_to_dataset:
        n_channels = 1
        n_classes = 11
    if 'organcmnist' in path_to_dataset:
        n_channels = 1
        n_classes = 11
    if 'organsmnist' in path_to_dataset:
        n_channels = 1
        n_classes = 11
    if 'pneumoniamnist' in path_to_dataset:
        n_channels = 1
        n_classes = 2

    return supervised_training_loader, unsupervised_training_loader, validation_loader, n_classes, n_channels



def sample_z(batch_size, num_classes, device):
    # Sample z_c as one-hot encoded vectors
    indices = torch.randint(low=0, high=num_classes, size=(batch_size,)).to(device)
    z_c = F.one_hot(indices, num_classes=num_classes).float().to(device)


    return z_c, indices



def compute_gradient_penalty(D, real_samples, fake_samples, device, debug = False):
    if debug:
        Tensor = torch.FloatTensor
    else:
        Tensor = torch.cuda.FloatTensor
    """Calculates the gradient penalty loss for WGAN-GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0),1,1,1))).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha*real_samples + ((1-alpha)*fake_samples)).requires_grad_(True) # requires_grad inplace
    d_interpolates, _ = D(interpolates) # adv_info, cls_info = discriminator(interpolated image)
    fake = torch.Tensor(np.ones(d_interpolates.shape)).to(device)
    # Get gradient w.r.t interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0),-1)
    gradient_penalty = ((gradients.norm(2, dim=1)-1)**2).mean()
    return gradient_penalty


def one_hot_encode(labels, num_classes):
    return F.one_hot(labels, num_classes)



class ImprovedEnsemble:
    def __init__(self, n_classes, temp=0.5):
        self.n_classes = n_classes
        self.temp = temp  # Temperature for sharpening predictions
        self.momentum = 0.99  # For EMA of predictions
        self.ema_predictions = None

    def time_ensemble(self, current_predictions, step):
        """Temporal ensembling using EMA of predictions"""
        avg_predictions = torch.mean(torch.stack(current_predictions), dim=0)

        if self.ema_predictions is None:
            self.ema_predictions = avg_predictions
        else:
            self.ema_predictions = self.momentum * self.ema_predictions + (1 - self.momentum) * avg_predictions

        return self.ema_predictions

    def weighted_voting(self, logits_list, confidences):
        """
        Weight predictions by model confidence
        confidences: list of confidence scores for each model
        """
        weighted_preds = []
        for logits, confidence in zip(logits_list, confidences):
            probs = F.softmax(logits / self.temp, dim=1)
            # Expand confidence to match probs shape
            confidence = confidence.unsqueeze(1).expand_as(probs)
            weighted_preds.append(probs * confidence)

        return torch.stack(weighted_preds).mean(dim=0)

    def get_model_confidence(self, logits):
        """Calculate model confidence based on prediction entropy"""
        probs = F.softmax(logits / self.temp, dim=1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
        max_entropy = torch.log(torch.tensor(self.n_classes, dtype=torch.float))
        confidence = 1 - (entropy / max_entropy)
        return confidence

    def generate_pseudo_labels(self, logits_list, percentile):
        """
        Generate pseudo-labels using multiple ensembling techniques
        """
        batch_size = logits_list[0].size(0)

        # 1. Get individual model confidences
        confidences = [self.get_model_confidence(logits) for logits in logits_list]

        # 2. Weighted voting based on model confidence
        weighted_preds = self.weighted_voting(logits_list, confidences)

        # 3. Add temporal ensemble predictions if available
        if self.ema_predictions is not None:
            weighted_preds = 0.7 * weighted_preds + 0.3 * self.ema_predictions

        # 4. Update temporal ensemble
        self.time_ensemble(logits_list, None)

        # 5. Generate pseudo-labels with confidence thresholding
        max_probs, pseudo_labels = torch.max(weighted_preds, dim=1)

        # Confidence thresholding
        threshold = torch.quantile(max_probs, percentile)
        confidence_mask = max_probs >= threshold

        # Convert to one-hot
        pseudo_labels_one_hot = torch.zeros(batch_size, self.n_classes).to(logits_list[0].device)
        pseudo_labels_one_hot[torch.arange(batch_size), pseudo_labels] = 1

        return pseudo_labels_one_hot, confidence_mask

def improved_soft_voting(logits_list, n_classes, percentile, device):
    """
    Wrapper function for the improved ensemble
    """
    ensemble = ImprovedEnsemble(n_classes)
    return ensemble.generate_pseudo_labels(logits_list, percentile)




def combine_logits(logits):
    sum_predictions = None
    for logit in logits:
        if sum_predictions is None:
            sum_predictions = logit
        else:
            sum_predictions += logit

    avg_predictions = sum_predictions / len(logits)
    return avg_predictions

def convert_to_one_hot(pred):
    predicted_classes = torch.argmax(pred, dim=1)
    one_hot_encoded = torch.zeros_like(pred)
    one_hot_encoded.scatter_(1, predicted_classes.unsqueeze(1), 1)
    return one_hot_encoded

def check_channel_dim(imgs):
    if imgs.size(1) == 1:
        imgs = imgs.repeat(1, 3, 1, 1)
    return imgs



def save_model(model, optimizer, best_acc, checkpoint_path):
    """
        Saves the PyTorch model's state_dict and the optimizer's state_dict.

        Args:
            model (torch.nn.Module): The PyTorch model to save.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            checkpoint_path (str): The path where the checkpoint will be saved.
        """

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc
        }, checkpoint_path)

def get_checkpoint_path():
    curr_work_dir = os.getcwd()
    checkpoint_path = os.path.join(curr_work_dir, 'checkpoints')
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    return checkpoint_path

def get_dataset_path(flag, percentage=0.1):
    if percentage == 0.1:
        dataset_folder = flag + '_10'
    elif percentage == 0.05:
        dataset_folder = flag + '_5'
    elif percentage == 0.01:
        dataset_folder = flag + '_1'
    elif percentage == 0.005:
        dataset_folder = flag + '_0.5'
    elif percentage == 0:
        dataset_folder = flag + '_oneshot'
    elif percentage == 2:
        dataset_folder = flag + '_twoshot'
    elif percentage == 3:
        dataset_folder = flag + '_threeshot'
    elif percentage == 4:
        dataset_folder = flag + '_4shot'
    elif percentage == 5:
        dataset_folder = flag + '_5shot'
    elif percentage == 6:
        dataset_folder = flag + '_6shot'
    elif percentage == 7:
        dataset_folder = flag + '_7shot'
    elif percentage == 8:
        dataset_folder = flag + '_8shot'
    elif percentage == 9:
        dataset_folder = flag + '_9shot'
    elif percentage == 10:
        dataset_folder = flag + '_10shot'
    elif percentage == 12:
        dataset_folder = flag + '_12shot'
    elif percentage == 14:
        dataset_folder = flag + '_14shot'
    elif percentage == 16:
        dataset_folder = flag + '_16shot'
    elif percentage == 18:
        dataset_folder = flag + '_18shot'
    elif percentage == 20:
        dataset_folder = flag + '_20shot'
    elif percentage == 50:
        dataset_folder = flag + '_50shot'
    elif percentage == 100:
        dataset_folder = flag + '_100shot'
    curr_work_dir = os.getcwd()
    dataset_path = os.path.join(curr_work_dir, 'dataset', dataset_folder)
    return dataset_path



def log_losses_mlflow(epoch, losses, flag):
    mlflow.log_metric(f"{flag} - Generator Loss", losses.generator_epoch_loss, epoch)
    mlflow.log_metric(f"{flag} - Discriminator Loss", losses.discriminator_epoch_loss, epoch)
    mlflow.log_metric(f"{flag} - Classifier Loss", losses.classifier_epoch_loss, epoch)


def log_classification_metrics_mlflow(epoch, classification_metrics):
    mlflow.log_metric("accuracy generator", classification_metrics['accuracy_generator'], epoch,)
    mlflow.log_metric("accuracy discriminator", classification_metrics['accuracy_discriminator'], epoch, )
    mlflow.log_metric("accuracy classifier", classification_metrics['accuracy_classifier'], epoch, )
    mlflow.log_metric("f1 generator", classification_metrics['f1_generator'], epoch, )
    mlflow.log_metric("f1 discriminator", classification_metrics['f1_discriminator'], epoch)
    mlflow.log_metric("f1 classifier", classification_metrics['f1_classifier'], epoch, )
    mlflow.log_metric("precision generator", classification_metrics['precision_generator'], epoch, )
    mlflow.log_metric("precision discriminator", classification_metrics['precision_discriminator'], epoch, )
    mlflow.log_metric("precision classifier", classification_metrics['precision_classifier'], epoch, )
    mlflow.log_metric("recall generator", classification_metrics['recall_generator'], epoch, )
    mlflow.log_metric("recall discriminator", classification_metrics['recall_discriminator'], epoch, )
    mlflow.log_metric("recall classifier", classification_metrics['recall_classifier'], epoch, )



class RealismValidator:
    def __init__(self, discriminator):
        """
        Args:
            discriminator: The WGAN discriminator that outputs (realism_score, class_prediction)
        """
        self.discriminator = discriminator
        self.discriminator.eval()
        self.realism_threshold = None

    def compute_threshold_from_real(self, real_images, margin_factor=1.2):
        """
        Compute threshold from real images using the maximum discriminator score
        (worst real image score) plus a margin factor.

        Args:
            real_images: Batch of real images
            margin_factor: Multiply the max score by this factor to allow more flexibility
        """
        with torch.no_grad():
            scores = []
            for img in real_images:
                score, _ = self.discriminator(img.unsqueeze(0))
                scores.append(score.mean().item())

            # Use maximum score instead of average
            max_real_score = max(scores)
            self.realism_threshold = max_real_score #* margin_factor

            # Optional: print some statistics about the real scores
            print(f"Real images score statistics:")
            print(f"Max score (threshold): {max_real_score:.4f}")
            print(f"Min score: {min(scores):.4f}")
            print(f"Mean score: {sum(scores) / len(scores):.4f}")
            print(f"Final threshold (with margin): {self.realism_threshold:.4f}")

            return self.realism_threshold

    def compute_realism_score(self, image):
        """Returns tuple of (realism_score, predicted_class)"""
        with torch.no_grad():
            score, pred_class = self.discriminator(image)
            return score.mean().item(), pred_class

    def is_realistic_enough(self, image, true_class):
        """
        Check if an image is realistic enough based on:
        1. Score being below threshold (more WGAN-like)
        2. Predicted class matching true class

        Args:
            image: The image to check
            true_class: The ground truth class

        Returns:
            tuple: (is_realistic, score, matches_class)
        """
        if self.realism_threshold is None:
            raise ValueError("Must compute threshold from real images first!")

        score, pred_class = self.compute_realism_score(image)

        # For WGAN, lower scores are better (more realistic)
        passes_threshold = score < self.realism_threshold

        # Check if predicted class matches ground truth
        pred_class = pred_class.argmax(dim=1)
        true_class = true_class.argmax(dim=1)
        matches_class = (pred_class == true_class).all()

        return passes_threshold, score, matches_class


def filter_generated_images(generator, discriminator, real_images, num_images, n_classes, device):
    """
    Generate images and filter them based on WGAN discriminator scores and class predictions.

    Args:
        generator: The trained generator model
        discriminator: The trained discriminator model
        real_images: Batch of real images to compute threshold from
        num_images: Number of realistic images to collect
        n_classes: Number of classes
        device: Device to run computation on

    Returns:
        tuple: (filtered_images, filtered_scores, filtered_classes)
    """
    validator = RealismValidator(discriminator)

    # First compute threshold from real images
    threshold = validator.compute_threshold_from_real(real_images)

    filtered_images = []
    filtered_scores = []
    filtered_classes = []

    total_attempted = 0

    with torch.no_grad():
        while len(filtered_images) < num_images:
            # Generate a batch of images
            zc, sampled_labels = sample_z(real_images.size(0), num_classes=n_classes, device=device)
            fake_images = generator(real_images, z=zc)

            total_attempted += len(fake_images)

            # Check realism for each image
            for idx, img in enumerate(fake_images):
                is_realistic, score, matches_class = validator.is_realistic_enough(
                    img.unsqueeze(0),
                    zc[idx].unsqueeze(0)
                )

                if is_realistic:  # Only keep if passes both threshold and class checks
                    filtered_images.append(img)
                    filtered_scores.append(score)
                    filtered_classes.append(zc[idx])

                if len(filtered_images) >= num_images:
                    break

    # Print statistics about the filtering process
    acceptance_rate = (len(filtered_images) / total_attempted) * 100
    print(f"\nFiltering statistics:")
    print(f"Total images attempted: {total_attempted}")
    print(f"Images accepted: {len(filtered_images)}")
    print(f"Acceptance rate: {acceptance_rate:.2f}%")

    return torch.stack(filtered_images), filtered_scores, torch.stack(filtered_classes)