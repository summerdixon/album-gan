import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from utils import normalize_embedding

def build_genre_to_idx(data_frame: pd.DataFrame) -> dict[str, int]:
    genres = sorted(data_frame["genre"].dropna().astype(str).unique().tolist())
    return {genre: idx for idx, genre in enumerate(genres)}


class AlbumGANDataset(Dataset):
    def __init__(self, csv_file, img_dir, embeddings_file, genre_to_idx=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.all_embeddings = torch.load(embeddings_file)
        self.genre_to_idx = genre_to_idx or build_genre_to_idx(self.data_frame)
        self.num_genres = len(self.genre_to_idx)
        
        # generator outputs images between -1 and 1, so we normalize our real images to match
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
        ])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        track_id = str(row['deezer_id'])
        img_path = os.path.join(self.img_dir, f"{track_id}.jpg")
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        audio_embedding = torch.as_tensor(self.all_embeddings[track_id]).float()
        genre_label = torch.tensor(self.genre_to_idx[str(row['genre'])], dtype=torch.long)
        return image, audio_embedding, genre_label

def get_dataloader(csv_path, img_dir, emb_path, batch_size=32, num_workers=2, genre_to_idx=None):
    dataset = AlbumGANDataset(
        csv_file=csv_path,
        img_dir=img_dir,
        embeddings_file=emb_path,
        genre_to_idx=genre_to_idx,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader

if __name__ == "__main__":
    test_loader = get_dataloader(
        csv_path="../data/cleaned_dataset.csv", 
        img_dir="../data/images", 
        audio_emb_path="../data/embeddings/audio_embeddings.pt",
    )
    
    # test getting one batch to prove it works
    imgs, audio_embs, genre_labels = next(iter(test_loader))
    print(f"Image shape: {imgs.shape}, Audio shape: {audio_embs.shape}, Genre shape: {genre_labels.shape}")