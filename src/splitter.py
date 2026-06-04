import os
import random
import shutil

class DatasetSplitter:
    def __init__(self, val_size=10, seed=777):
        self.val_size = val_size
        self.seed = seed

    def split(self, input_dir, train_dir, val_dir):
        # Initialize directories
        for d in [train_dir, val_dir]:
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)

        # Get all clips
        all_clips = [f for f in os.listdir(input_dir) if f.endswith('.wav')]

        if len(all_clips) < self.val_size:
            print(f"❌ Total clips ({len(all_clips)}) is less than validation size ({self.val_size}). Cannot split.")
            return

        # Shuffle and split
        random.seed(self.seed)
        random.shuffle(all_clips)

        val_set = all_clips[:self.val_size]
        train_set = all_clips[self.val_size:]

        print("📦 Deploying files to final directories...")
        for f in val_set:
            shutil.copy(os.path.join(input_dir, f), os.path.join(val_dir, f))
        for f in train_set:
            shutil.copy(os.path.join(input_dir, f), os.path.join(train_dir, f))

        print(f"⭐ [Final Report] ⭐")
        print(f" - Train Dataset      : {len(train_set)} clips -> {train_dir}")
        print(f" - Validation Dataset : {len(val_set)} clips (exactly {self.val_size}) -> {val_dir}")
