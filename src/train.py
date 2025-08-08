import argparse

# AI - Libraries
import torch
import torchvision.transforms as transforms
import torch.utils.data as data

# Other - Libraries
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# Internal Libraries
from training_utils.losses import *
from training_utils.utils import *

import mlflow


def supervised_training_loop(models, supervised_train_loader, losses, optimizers, n_classes, device):
    losses.reset_stored_loss()

    few_shot_losses = FewShotLosses(n_classes=n_classes)

    # Unpack models
    generator = models['generator'].train()
    discriminator = models['discriminator'].train()
    classifier = models['classifier'].train()

    # Unpack optimizers
    optimizer_G = optimizers['generator']
    optimizer_D = optimizers['discriminator']
    optimizer_C = optimizers['classifier']

    for inputs, labels in tqdm(supervised_train_loader):
        # Move inputs and labels to the device
        inputs = inputs.to(device)
        labels_one_hot = one_hot_encode(labels, num_classes=n_classes).to(device).float()

        # Zero the gradients
        optimizer_G.zero_grad()
        optimizer_D.zero_grad()
        optimizer_C.zero_grad()

        # Forward pass
        predictions_generator = generator(inputs, mode="classifier")
        _, predictions_discriminator = discriminator(inputs)
        predictions_classifier = classifier(inputs)

        # Compute probabilities once and detach for mutual learning
        with torch.no_grad():
            prob_G = F.softmax(predictions_generator.detach(), dim=1)
            prob_D = F.softmax(predictions_discriminator.detach(), dim=1)
            prob_C = F.softmax(predictions_classifier.detach(), dim=1)

        # 1. Prototype loss for each model
        proto_loss_G = few_shot_losses.prototype_loss(predictions_generator, labels_one_hot, inputs)
        proto_loss_D = few_shot_losses.prototype_loss(predictions_discriminator, labels_one_hot, inputs)
        proto_loss_C = few_shot_losses.prototype_loss(predictions_classifier, labels_one_hot, inputs)

        # 2. Mutual learning between models
        mutual_loss = few_shot_losses.triple_mutual_learning_loss(
            prob_G,
            prob_D,
            prob_C,
            labels_one_hot
        )


        # 3. Entropy minimization for confident predictions
        entropy_loss_G = few_shot_losses.entropy_minimization_loss(predictions_generator)
        entropy_loss_D = few_shot_losses.entropy_minimization_loss(predictions_discriminator)
        entropy_loss_C = few_shot_losses.entropy_minimization_loss(predictions_classifier)

        # 4. Mixup for data augmentation
        mixup_loss_G = few_shot_losses.mixup_loss_generator(generator, inputs, labels_one_hot)
        mixup_loss_D = few_shot_losses.mixup_loss_discriminator(discriminator, inputs, labels_one_hot)
        mixup_loss_C = few_shot_losses.mixup_loss_classifier(classifier, inputs, labels_one_hot)

        # Compute losses
        supervised_classification_loss_G = proto_loss_G + 0.1 * mutual_loss + entropy_loss_G + mixup_loss_G #+ losses.criterion_classification(predictions_generator, labels_one_hot)
        supervised_classification_loss_D = proto_loss_D + 0.1 * mutual_loss + entropy_loss_D + mixup_loss_D + losses.criterion_classification(predictions_discriminator, labels_one_hot)
        supervised_classification_loss_C = proto_loss_C + 0.1 * mutual_loss + entropy_loss_C + mixup_loss_C + losses.criterion_classification(predictions_classifier, labels_one_hot)


        # Backward pass
        supervised_classification_loss_G.backward()
        supervised_classification_loss_D.backward()
        supervised_classification_loss_C.backward()

        # Update weights
        optimizer_G.step()
        optimizer_D.step()
        optimizer_C.step()

        # Update cumulative loss
        losses.cum_generator_loss += supervised_classification_loss_G.item()
        losses.cum_discriminator_loss += supervised_classification_loss_D.item()
        losses.cum_classifier_loss += supervised_classification_loss_C.item()
        losses.batches += 1

    losses.compute_epoch_loss()

def unsupervised_training_loop(models, unsupervised_train_loader, losses, optimizers, n_classes, percentile, device, mec):
    losses.reset_stored_loss()
    # Unpack models
    generator = models['generator'].train()
    discriminator = models['discriminator'].train()
    classifier = models['classifier'].train()

    # Unpack optimizers
    optimizer_G = optimizers['generator']
    optimizer_D = optimizers['discriminator']
    optimizer_C = optimizers['classifier']

    for inputs, _ in tqdm(unsupervised_train_loader):
        imgs = check_channel_dim(inputs).to(device)

        # classify the inputs
        pseudo_logits_classifier = classifier(imgs)
        _, pseudo_logits_discriminator = discriminator(imgs)
        #pseudo_logits_generator = generator(imgs, mode="classifier")



        pseudo_logits = [pseudo_logits_classifier, pseudo_logits_discriminator]

        pseudo_gt, confidence_mask = improved_soft_voting(pseudo_logits, n_classes, percentile, device)

        # Filter inputs based on confidence threshold
        confident_imgs = imgs[confidence_mask]
        confident_pseudo_gt = pseudo_gt[confidence_mask]

        if confident_imgs.size(0) == 0:
            continue  # Skip this batch if no confident predictions


        pseudo_one_hot = convert_to_one_hot(pseudo_gt).to(device)
        pseudo_labels = torch.argmax(pseudo_one_hot, dim=1).to(device)

        # Zero the gradients
        optimizer_G.zero_grad()
        optimizer_D.zero_grad()
        optimizer_C.zero_grad()

        # Forward pass Discriminator

        # Real images
        real_validity, pred_cls = discriminator(imgs)

        # Fake images
        zc, sampled_labels = sample_z(imgs.size(0), num_classes=n_classes, device=device)
        gen_imgs = generator(imgs, z=zc)
        fake_validity, _ = discriminator(gen_imgs.detach())

        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, imgs.data, gen_imgs.data, device)

        # Adversarial loss
        loss_D_adv = -torch.mean(real_validity) + torch.mean(fake_validity) + 10 * gradient_penalty

        # Classification loss
        loss_D_cls = losses.criterion_classification(pred_cls, pseudo_one_hot)

        # Total discriminator loss
        loss_D = loss_D_adv + loss_D_cls

        # Backward pass Discriminator
        loss_D.backward()
        optimizer_D.step()

        # Forward pass Generator
        gen_imgs = generator(imgs, z=zc)
        recov_imgs = generator(gen_imgs, z=pseudo_one_hot)

        # Discriminator evaluates translated images
        fake_validity, pred_cls_D = discriminator(gen_imgs)
        pred_cls_G = generator(gen_imgs, mode="classifier")


        # Adversarial loss
        loss_G_adv = -torch.mean(fake_validity)

        # Classification loss on gen imgs
        loss_G_cls_D = losses.criterion_classification(pred_cls_D, zc)
        loss_G_cls_G = losses.criterion_classification(pred_cls_G, zc)



        # Reconstruction loss
        loss_G_rec = losses.criterion_cycle(recov_imgs, imgs)

        # Tota loss
        loss_G = loss_G_adv + loss_G_cls_D + loss_G_cls_G + 10 * loss_G_rec #+ loss_G_cls_rec

        # Backward pass Generator
        loss_G.backward()
        optimizer_G.step()

        # Forward pass Classifier
        gen_imgs = check_channel_dim(generator(imgs, z=zc))

        # classify the generated images
        generated_logits_classifier = classifier(gen_imgs.detach())

        # Compute losses
        loss_C = losses.criterion_classification(generated_logits_classifier, zc)


        # Backward pass Classifier
        loss_C.backward()
        optimizer_C.step()

        # Update cumulative loss
        losses.cum_generator_loss += loss_G.item()
        losses.cum_discriminator_loss += loss_D.item()
        losses.cum_classifier_loss += loss_C.item()
        losses.batches += 1

    losses.compute_epoch_loss()


def test_loop(models, test_loader, losses, n_classes, device):
    losses.reset_stored_loss()

    # Unpack models
    generator = models['generator'].eval()
    discriminator = models['discriminator'].eval()
    classifier = models['classifier'].eval()

    all_labels = []
    all_predictions_generator = []
    all_predictions_discriminator = []
    all_predictions_classifier = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            # Move inputs and labels to the device
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels_one_hot = one_hot_encode(labels, num_classes=n_classes).to(device).float()

            # Forward pass
            predictions_generator = generator(inputs, mode="classifier")
            _, predictions_discriminator = discriminator(inputs)
            predictions_classifier = classifier(inputs)

            # Compute losses
            supervised_classification_loss_G = losses.criterion_classification(predictions_generator, labels_one_hot)
            supervised_classification_loss_D = losses.criterion_classification(predictions_discriminator, labels_one_hot)
            supervised_classification_loss_C = losses.criterion_classification(predictions_classifier, labels_one_hot)

            # Update cumulative loss
            losses.cum_generator_loss += supervised_classification_loss_G.item()
            losses.cum_discriminator_loss += supervised_classification_loss_D.item()
            losses.cum_classifier_loss += supervised_classification_loss_C.item()
            losses.batches += 1

            # store labels & predictions for classification evaluation_utils
            all_labels.extend(labels.cpu().numpy())
            all_predictions_generator.extend(torch.argmax(predictions_generator, dim=1).cpu().numpy())
            all_predictions_discriminator.extend(torch.argmax(predictions_discriminator, dim=1).cpu().numpy())
            all_predictions_classifier.extend(torch.argmax(predictions_classifier, dim=1).cpu().numpy())

    # Compute epoch loss
    losses.compute_epoch_loss()

    # Compute classification metrics
    accuracy_generator = accuracy_score(all_labels, all_predictions_generator)
    accuracy_discriminator = accuracy_score(all_labels, all_predictions_discriminator)
    accuracy_classifier = accuracy_score(all_labels, all_predictions_classifier)

    precision_generator = precision_score(all_labels, all_predictions_generator, average='weighted')
    precision_discriminator = precision_score(all_labels, all_predictions_discriminator, average='weighted')
    precision_classifier = precision_score(all_labels, all_predictions_classifier, average='weighted')

    recall_generator = recall_score(all_labels, all_predictions_generator, average='weighted')
    recall_discriminator = recall_score(all_labels, all_predictions_discriminator, average='weighted')
    recall_classifier = recall_score(all_labels, all_predictions_classifier, average='weighted')

    f1_generator = f1_score(all_labels, all_predictions_generator, average='weighted')
    f1_discriminator = f1_score(all_labels, all_predictions_discriminator, average='weighted')
    f1_classifier = f1_score(all_labels, all_predictions_classifier, average='weighted')

    confusion_matrix_generator = confusion_matrix(all_labels, all_predictions_generator)
    confusion_matrix_discriminator = confusion_matrix(all_labels, all_predictions_discriminator)
    confusion_matrix_classifier = confusion_matrix(all_labels, all_predictions_classifier)


    classification_metrics = {
        'accuracy_generator': accuracy_generator,
        'accuracy_discriminator': accuracy_discriminator,
        'accuracy_classifier': accuracy_classifier,
        'precision_generator': precision_generator,
        'precision_discriminator': precision_discriminator,
        'precision_classifier': precision_classifier,
        'recall_generator': recall_generator,
        'recall_discriminator': recall_discriminator,
        'recall_classifier': recall_classifier,
        'f1_generator': f1_generator,
        'f1_discriminator': f1_discriminator,
        'f1_classifier': f1_classifier,
        'confusion_matrix_generator': confusion_matrix_generator,
        'confusion_matrix_discriminator': confusion_matrix_discriminator,
        'confusion_matrix_classifier': confusion_matrix_classifier
    }


    losses.compute_epoch_loss()

    return classification_metrics



def main(args):

    # Initialize wandb

    run_name = f'{args.exp_id}_{args.project_id}_DT_{args.data_flag}'

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define checkpoint path
    checkpoint_path = get_checkpoint_path()

    checkpoint_disc = os.path.join(checkpoint_path, f'discriminator_{args.exp_id}_{args.project_id}_DT_{args.data_flag}.pth')
    checkpoint_gen = os.path.join(checkpoint_path, f'generator_{args.exp_id}_{args.project_id}_DT_{args.data_flag}.pth')
    checkpoint_cls = os.path.join(checkpoint_path, f'classifier_{args.exp_id}_{args.project_id}_DT_{args.data_flag}.pth')

    # Load the dataset
    flag = os.path.join(args.path_to_dt, args.data_flag)
    dataset_path = get_dataset_path(flag, args.percentage_supervised)
    supervised_train_loader, unsupervised_train_loader, test_loader, n_classes, n_channels = load_dataset(dataset_path, args.batch_size)

    # Define the models
    models = initialize_model_and_weight((3, 128, 128), n_classes, device)

    # Define the losses
    losses = Losses()

    # Define the optimizers
    optimizers = initialize_optimzers(models, args.lr, args.betas)

    # Initialize best accuracy
    best_gen_accuracy = 0.0
    best_disc_accuracy = 0.0
    best_cls_accuracy = 0.0
    with mlflow.start_run():
        mlflow.log_param("Run Name", run_name)
        mlflow.log_param("n_unsupervised", args.n_unsupervised)
        mlflow.set_experiment(args.project_id)

        for epoch in range(args.epochs):
            print("[INFO]: Starting Supervised Training ...")
            supervised_training_loop(models, supervised_train_loader, losses, optimizers, n_classes, device)
            print("[INFO]: logging losses from supervised training ...")
            log_losses_mlflow(epoch, losses, "Supervised")

            if args.n_unsupervised != 0:
                if epoch % args.n_unsupervised == 0:
                    print("[INFO]: Starting Unsupervised Training ...")
                    unsupervised_training_loop(models, unsupervised_train_loader, losses, optimizers, n_classes, args.percentile, device, mec=False)
                    print("[INFO]: logging losses from unsupervised training ...")
                    log_losses_mlflow(epoch, losses, "Unsupervised")
            else:
                print("[INFO]: Skipping Unsupervised training")

            print("[INFO]: Starting Testing ...")
            classification_metrics = test_loop(models, test_loader, losses, n_classes, device)
            print("[INFO]: logging losses from testing ...")
            log_losses_mlflow(epoch, losses, "Testing")
            print("[INFO]: logging classification metrics ...")
            log_classification_metrics_mlflow(epoch, classification_metrics)



            # Save the best model
            if classification_metrics['accuracy_generator'] > best_gen_accuracy:
                best_gen_accuracy = classification_metrics['accuracy_generator']
                save_model(models['generator'], optimizers['generator'], best_gen_accuracy, checkpoint_gen)

            if classification_metrics['accuracy_discriminator'] > best_disc_accuracy:
                best_disc_accuracy = classification_metrics['accuracy_discriminator']
                save_model(models['discriminator'], optimizers['discriminator'], best_disc_accuracy, checkpoint_disc)

            if classification_metrics['accuracy_classifier'] > best_cls_accuracy:
                best_cls_accuracy = classification_metrics['accuracy_classifier']
                save_model(models['classifier'], optimizers['classifier'], best_cls_accuracy, checkpoint_cls)









if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training configuration')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--betas', type=tuple, default=(0.5, 0.999), help='Betas for Adam optimizer')
    parser.add_argument('--batch_size', type=int, default=24, help='Batch size')
    parser.add_argument('--data_flag', type=str, default='bloodmnist', help='Dataset flag')
    parser.add_argument('--path_to_dt', type=str, default='/mimer/NOBACKUP/groups/naiss2023-6-336/gmanni/dataset')
    parser.add_argument("--percentage_supervised", type=float, default=5, help="Percentage of supervised data")
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--load_checkpoint', type=bool, default=False, help='Load checkpoint flag')
    parser.add_argument('--project_id', type=str, default='SSCG', help='Project ID')
    parser.add_argument('--exp_id', type=str, default='val_5shot', help='Experiment ID')
    parser.add_argument('--percentile', type=float, default=0.75, help='Percentile for confidence threshold')
    parser.add_argument('--n_unsupervised', type=int, default=5, help='Number of times unsupervised training is performed')



    args = parser.parse_args()

    main(args)










