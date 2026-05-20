import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from utils import normalize_embedding

class AlbumGANDataset(Dataset):
    def __init__(self, csv_file, img_dir, audio_embeddings_file, text_embeddings_file):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.audio_embeddings = torch.load(audio_embeddings_file, map_location="cpu")
        self.text_embeddings = torch.load(text_embeddings_file, map_location="cpu")
        
        # generator outputs images between -1 and 1, so we normalize our real images to match
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
        ])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        track_id = str(self.data_frame.iloc[idx]['deezer_id'])
        img_path = os.path.join(self.img_dir, f"{track_id}.jpg")
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        audio_embedding = normalize_embedding(self.audio_embeddings[track_id])
        text_embedding = normalize_embedding(self.text_embeddings[track_id])

        combined_embedding = torch.stack([audio_embedding, text_embedding], dim=0)
        return image, combined_embedding

def get_dataloader(csv_path, img_dir, audio_emb_path, text_emb_path, batch_size=32, num_workers=2):
    dataset = AlbumGANDataset(
        csv_file=csv_path,
        img_dir=img_dir,
        audio_embeddings_file=audio_emb_path,
        text_embeddings_file=text_emb_path,
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
        text_emb_path="../data/embeddings/text_embeddings.pt",
    )
    
    # test getting one batch to prove it works
    imgs, audio_embs = next(iter(test_loader))
    print(f"Image shape: {imgs.shape}, Audio shape: {audio_embs.shape}")