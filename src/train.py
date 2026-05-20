import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from data import get_dataloader
from torchvision.utils import make_grid, save_image
from model import Generator, Discriminator
from utils import normalize_embedding, resize_image_tensor

def train():

    # Configuration
    epochs = 80
    test_interval = 5  # Test every 5 epochs
    test_set_size = 20  # Number of tracks for testing (15-30 range)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Loss
    criterion = nn.BCELoss()

    # Data paths
    csv_path = "../data/cleaned_dataset.csv"
    img_dir = "../data/images"
    audio_emb_path = "../data/embeddings/audio_embeddings.pt"
    batch_size = 32

    # Load the already-stacked conditioning embeddings for sampling
    all_audio_embeddings = torch.load(audio_emb_path, map_location="cpu")

    def load_embedding(embedding_dict, track_id):
        if track_id in embedding_dict:
            return embedding_dict[track_id]

        try:
            numeric_track_id = int(track_id)
        except ValueError:
            return None

        return embedding_dict.get(numeric_track_id)

    def combine_embeddings(track_id):
        audio_embedding = load_embedding(all_audio_embeddings, track_id)

        if audio_embedding is None:
            return None

        return normalize_embedding(audio_embedding)

    # Samples output directory
    samples_dir = "../samples"
    os.makedirs(samples_dir, exist_ok=True)
    sample_image_size = (256, 256)

    # Load and split the dataset
    data_frame = pd.read_csv(csv_path)
    train_df, test_df = train_test_split(
        data_frame,
        test_size=test_set_size,
        random_state=42
    )

    print(f"Total samples: {len(data_frame)}")
    print(f"Training samples: {len(train_df)}")
    print(f"Testing samples: {len(test_df)}")

    # Save train/test splits for reproducibility
    train_df.to_csv("../data/train_dataset.csv", index=False)
    test_df.to_csv("../data/test_dataset.csv", index=False)

    # Create dataloaders
    train_loader = get_dataloader(
        csv_path="../data/train_dataset.csv",
        img_dir=img_dir,
        audio_emb_path=audio_emb_path,
        batch_size=batch_size
    )

    test_loader = get_dataloader(
        csv_path="../data/test_dataset.csv",
        img_dir=img_dir,
        audio_emb_path=audio_emb_path,
        batch_size=batch_size
    )   

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Initialize models
    gen = Generator(embedding_dim=512, num_embeddings=2).to(device)
    disc = Discriminator(embedding_dim=512, num_embeddings=2).to(device)

    # Optimizers (Adam)
    learning_rate = 0.0002
    betas = (0.5, 0.999)
    optimizer_G = optim.Adam(gen.parameters(), lr=learning_rate, betas=betas)
    optimizer_D = optim.Adam(disc.parameters(), lr=learning_rate, betas=betas)

    print(f"Using device: {device}")

    # Checkpointing weights
    weights_dir = "../weights"
    os.makedirs(weights_dir, exist_ok=True)
    save_interval = 10  # save every N epochs
    checkpoint_last = os.path.join(weights_dir, "checkpoint_last.pth")
    # Set this to a path to resume from, or leave as None
    resume_from = checkpoint_last

    # Try to resume if requested
    start_epoch = 0
    if resume_from is not None and os.path.exists(resume_from):
        ckpt = torch.load(resume_from, map_location=device)
        gen.load_state_dict(ckpt["gen_state"])
        disc.load_state_dict(ckpt["disc_state"])
        optimizer_G.load_state_dict(ckpt["optG_state"])
        optimizer_D.load_state_dict(ckpt["optD_state"])
        last_epoch = ckpt.get("epoch", -1)
        start_epoch = last_epoch + 1
        print(f"Resumed from checkpoint {resume_from}, starting at epoch {start_epoch+1}")
    else:
        print("Could not resume from requested checkpoint, starting from beginning")

    # Training loop structure
    for epoch in range(start_epoch, epochs):
        gen.train()
        disc.train()

        epoch_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Train", leave=False)
        for batch_idx, (images, audio_embeddings) in enumerate(epoch_iter):
            images = images.to(device)
            audio_embeddings = audio_embeddings.to(device)

            batch_size_curr = images.size(0)

            # Labels
            real_labels = torch.ones(batch_size_curr, 1, device=device, dtype=torch.float)
            fake_labels = torch.zeros(batch_size_curr, 1, device=device, dtype=torch.float)

            # -----------------
            #  Train Discriminator
            # -----------------
            optimizer_D.zero_grad()

            # Real images
            outputs_real = disc(images, audio_embeddings)
            loss_real = criterion(outputs_real, real_labels)

            # Fake images
            fake_images = gen(audio_embeddings)
            outputs_fake = disc(fake_images.detach(), audio_embeddings)
            loss_fake = criterion(outputs_fake, fake_labels)

            loss_D = (loss_real + loss_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            outputs_fake_for_G = disc(fake_images, audio_embeddings)
            loss_G = criterion(outputs_fake_for_G, real_labels)
            loss_G.backward()
            optimizer_G.step()

            epoch_iter.set_postfix({"loss_D": f"{loss_D.item():.4f}", "loss_G": f"{loss_G.item():.4f}"})

        # Testing / Evaluation phase every N epochs
        if (epoch + 1) % test_interval == 0:
            gen.eval()
            disc.eval()
            print(f"Epoch {epoch + 1}/{epochs} - Running evaluation on test set")
            with torch.no_grad():
                test_iter = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} - Test", leave=False)
                for batch_idx, (images, audio_embeddings) in enumerate(test_iter):
                    images = images.to(device)
                    audio_embeddings = audio_embeddings.to(device)

                    # Example eval: discriminator score on real images
                    outputs = disc(images, audio_embeddings)
                    avg_score = outputs.mean().item()
                    test_iter.set_postfix({"disc_real_avg": f"{avg_score:.4f}"})

                # Generate and save samples from test set embeddings
                # Use test_df ordering to get track ids
                sample_ids = [str(x) for x in test_df['deezer_id'].tolist()]
                # Build embedding tensor batch
                sample_pairs = []
                for tid in sample_ids:
                    emb = combine_embeddings(tid)
                    if emb is not None:
                        sample_pairs.append((tid, emb))

                if not sample_pairs:
                    raise RuntimeError("No matching conditioning embeddings found for the test samples.")

                sample_ids = [tid for tid, _ in sample_pairs]
                emb_list = [emb for _, emb in sample_pairs]
                emb_batch = torch.stack(emb_list).to(device)
                fake_images = gen(emb_batch)
                fake_images = resize_image_tensor(fake_images, size=sample_image_size)

                # Save each generated image named with track_id and epoch
                for i, tid in enumerate(sample_ids):
                    img_tensor = fake_images[i].unsqueeze(0)
                    grid = make_grid(img_tensor, nrow=1, normalize=True, value_range=(-1, 1))
                    out_path = os.path.join(samples_dir, f"{tid}_{epoch+1}.png")
                    save_image(grid, out_path)

            # Checkpoint saving every `save_interval` epochs
            if (epoch + 1) % save_interval == 0:
                ckpt = {
                    "epoch": epoch,
                    "gen_state": gen.state_dict(),
                    "disc_state": disc.state_dict(),
                    "optG_state": optimizer_G.state_dict(),
                    "optD_state": optimizer_D.state_dict(),
                }
                ckpt_path = os.path.join(weights_dir, f"checkpoint_epoch_{epoch+1}.pth")
                torch.save(ckpt, ckpt_path)
                torch.save(ckpt, checkpoint_last)
                print(f"Saved checkpoint: {ckpt_path}")

if __name__ == "__main__":
    train()