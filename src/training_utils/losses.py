# AI - Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature:float=0.1):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, feature_vectors, labels):
        # Normalize feature vectors
        feature_vectors_normalized = F.normalize(feature_vectors, p=2, dim=1)
        # Compute logits
        logits = torch.div(
            torch.matmul(
                feature_vectors_normalized, torch.transpose(feature_vectors_normalized, 0, 1)
            ),
            self.temperature,
        )
        return losses.NTXentLoss(temperature=0.07)(logits, torch.squeeze(labels))

class ClassLoss(nn.Module):
    def __init__(self):
        super(ClassLoss, self).__init__()

    def forward(self, logits, labels):
        return F.binary_cross_entropy_with_logits(logits, labels, reduction='sum') / logits.size(0)


class Losses:
    def __init__(self, temperature: float = 0.1):
        '''
        Constructor for the Losses class.
        :param temperature: temperature parameter for the supervised contrastive loss.
        '''
        # initialize the loss functions
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()
        self.criterion_supervised_contrastive = SupervisedContrastiveLoss(temperature)
        self.criterion_classification = ClassLoss()


        # initialize cumulative loss for each epoch
        self.cum_generator_loss = 0.0
        self.cum_discriminator_loss = 0.0
        self.cum_classifier_loss = 0.0

        # initialize epoch losses
        self.generator_epoch_loss = 0.0
        self.discriminator_epoch_loss = 0.0
        self.classifier_epoch_loss = 0.0

        self.batches = 0

    def reset_stored_loss(self):
        self.cum_generator_loss = 0.0
        self.cum_discriminator_loss = 0.0
        self.cum_classifier_loss = 0.0
        self.batches = 0

    def compute_epoch_loss(self):
        self.generator_epoch_loss = (self.cum_generator_loss/self.batches)
        self.discriminator_epoch_loss = (self.cum_discriminator_loss/self.batches)
        self.classifier_epoch_loss = (self.cum_classifier_loss/self.batches)



class FewShotLosses:
    def __init__(self, n_classes, temp=0.1):
        self.n_classes = n_classes
        self.temp = temp  # Temperature for softmax

    def prototype_loss(self, predictions, labels, inputs):
        """
        Creates class prototypes from the few examples and encourages
        predictions to be close to their class prototype
        """
        batch_size = predictions.size(0)

        # Convert predictions to probability distribution
        probs = F.softmax(predictions / self.temp, dim=1)

        # Create class prototypes by averaging embeddings of same class
        prototypes = torch.zeros(self.n_classes, predictions.size(1)).to(predictions.device)
        for c in range(self.n_classes):
            mask = (labels[:, c] == 1)
            if mask.sum() > 0:  # If we have examples of this class
                prototypes[c] = probs[mask].mean(0)

        # Calculate distance to prototypes
        expanded_prototypes = prototypes.unsqueeze(0).expand(batch_size, -1, -1)
        expanded_embeddings = probs.unsqueeze(1).expand(-1, self.n_classes, -1)

        # Calculate loss as negative log-likelihood of correct prototype
        distances = -((expanded_embeddings - expanded_prototypes) ** 2).sum(-1)
        loss = F.cross_entropy(distances, torch.argmax(labels, dim=1))

        return loss

    def triple_mutual_learning_loss(self, pred_G, pred_D, pred_C, labels):
        """
        Encourages all three models to agree on supervised samples
        while maintaining their uniqueness
        """
        # Convert all predictions to probabilities
        prob_G = F.softmax(pred_G / self.temp, dim=1)
        prob_D = F.softmax(pred_D / self.temp, dim=1)
        prob_C = F.softmax(pred_C / self.temp, dim=1)

        # Standard cross entropy for each model
        loss_sup = (
                F.cross_entropy(pred_G, torch.argmax(labels, dim=1)) +
                F.cross_entropy(pred_D, torch.argmax(labels, dim=1)) +
                F.cross_entropy(pred_C, torch.argmax(labels, dim=1))
        )

        # Mutual learning term
        loss_mutual = (
                F.kl_div(prob_G.log(), (prob_D + prob_C) / 2, reduction='batchmean') +
                F.kl_div(prob_D.log(), (prob_G + prob_C) / 2, reduction='batchmean') +
                F.kl_div(prob_C.log(), (prob_G + prob_D) / 2, reduction='batchmean')
        )

        return loss_sup + 0.1 * loss_mutual


    def entropy_minimization_loss(self, predictions):
        """
        Encourages confident predictions by minimizing entropy
        """
        probs = F.softmax(predictions, dim=1)
        log_probs = F.log_softmax(predictions, dim=1)
        entropy = -(probs * log_probs).sum(dim=1).mean()

        return 0.1 * entropy

    def mixup_loss_generator(self, generator, inputs, labels, alpha=0.2):
        lam = torch.distributions.Beta(alpha, alpha).sample().to(inputs.device)
        perm = torch.randperm(inputs.size(0))
        mixed_inputs = lam * inputs + (1 - lam) * inputs[perm]
        mixed_labels = lam * labels + (1 - lam) * labels[perm]
        predictions = generator(mixed_inputs, mode="classifier")
        return F.cross_entropy(predictions, mixed_labels)

    def mixup_loss_discriminator(self, discriminator, inputs, labels, alpha=0.2):
        lam = torch.distributions.Beta(alpha, alpha).sample().to(inputs.device)
        perm = torch.randperm(inputs.size(0))
        mixed_inputs = lam * inputs + (1 - lam) * inputs[perm]
        mixed_labels = lam * labels + (1 - lam) * labels[perm]
        _, predictions = discriminator(mixed_inputs)
        return F.cross_entropy(predictions, mixed_labels)

    def mixup_loss_classifier(self, classifier, inputs, labels, alpha=0.2):
        lam = torch.distributions.Beta(alpha, alpha).sample().to(inputs.device)
        perm = torch.randperm(inputs.size(0))
        mixed_inputs = lam * inputs + (1 - lam) * inputs[perm]
        mixed_labels = lam * labels + (1 - lam) * labels[perm]
        predictions = classifier(mixed_inputs)
        return F.cross_entropy(predictions, mixed_labels)

