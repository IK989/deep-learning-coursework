# generator.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import random
import copy # Used for deep copying augmented images

class ImageGenerator:
    """
    Implements a data generator that handles image loading, batch generation,
    resizing, shuffling, and synthetic data augmentation.
    """
    
    # Class dictionary mapping integer labels to names
    class_dict = {
        0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
        5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
    }

    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        """
        Initializes the ImageGenerator.
        
        Args:
            file_path: Path to the directory containing images.
            label_path: Path to the JSON file containing labels.
            batch_size: Number of images per batch.
            image_size: Desired image size [H, W, C].
            rotation, mirroring, shuffle: Optional bool flags for augmentation/shuffling.
        """
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle

        # Load labels JSON file
        with open(label_path, 'r') as f:
            self.labels = json.load(f)

        # Get all image filenames (e.g., '1.npy')
        self.image_files = sorted([f for f in os.listdir(file_path) if f.endswith('.npy')], 
                                  key=lambda f: int(os.path.splitext(f)[0]))
        self.num_images = len(self.image_files)

        # Create indices for the dataset
        self.indices = np.arange(self.num_images)
        if self.shuffle:
            np.random.shuffle(self.indices)

        self.current_index = 0
        self.epoch = 0

    def next(self):
        """
        Returns one batch of the provided dataset as a tuple (images, labels).
        (Task: Implement next())
        """
        if self.current_index + self.batch_size > self.num_images:
            # End of epoch: must wrap around and reset
            remaining = self.num_images - self.current_index
            
            # 1. Indices from the old epoch (the tail)
            batch_indices_tail = self.indices[self.current_index:] 

            # --- EPOCH RESET & RESHUFFLE FIX ---
            self.epoch += 1
            self.current_index = 0
            
            # FIX for testShuffleEpoch: Shuffle the indices array for the new epoch immediately
            if self.shuffle:
                np.random.shuffle(self.indices)
            
            # 2. Indices from the new epoch (the head)
            batch_indices_head = self.indices[:self.batch_size - remaining]
            
            # Combine
            batch_indices = np.concatenate((batch_indices_tail, batch_indices_head))
            self.current_index = len(batch_indices_head) # Update index for the start of the next full batch
        else:
            # Standard batch
            batch_indices = self.indices[self.current_index:self.current_index + self.batch_size]
            self.current_index += self.batch_size

        # Load and process data
        images = []
        labels = []

        for idx in batch_indices:
            file_name = self.image_files[idx]
            # Use np.load to read the .npy file
            img = np.load(os.path.join(self.file_path, file_name)) 

            # Resizing (allowed library function: skimage.transform.resize)
            # Do not confuse resizing with reshaping!
            img = resize(img, self.image_size, anti_aliasing=True)

            # Augment
            augmented_img = self._augment(img)
            
            # FIX for testResetIndex: The test requires that overlapping elements 
            # in consecutive batches are not the same Python object.
            images.append(copy.deepcopy(augmented_img)) 
            
            # Get label (key is filename without extension)
            label = self.labels[os.path.splitext(file_name)[0]]
            labels.append(label)

        # Labels must be returned as an array of integers (Requirement for testLabels)
        return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)

    def _augment(self, img):
        """Helper method to apply random rotation and mirroring."""
        
        # Random mirroring (horizontally is standard for np.fliplr)
        if self.mirroring and random.choice([True, False]):
            img = np.fliplr(img)
        
        # Random rotation by 90, 180, or 270 degrees
        if self.rotation:
            # random.choice([0, 1, 2, 3]) includes 0 degrees, which reduces augmentation effectiveness.
            # Use random.choice([1, 2, 3]) for 90, 180, 270 degrees as intended by the task/test logic.
            k = random.choice([1, 2, 3]) 
            img = np.rot90(img, k)

        return img

    def current_epoch(self):
        """Returns an integer of the current epoch."""
        return self.epoch

    def class_name(self, label):
        """Returns the class name that corresponds to the integer label."""
        return self.class_dict.get(label, "Unknown")

    def show(self):
        """Generates a batch using next() and plots it."""
        images, labels = self.next()
        
        # Determine grid size (assuming a two-row display for this large batch size)
        rows = 2
        cols = int(np.ceil(self.batch_size / rows))
        
        plt.figure(figsize=(cols * 2.5, rows * 3))
        
        for i in range(len(images)):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(images[i])
            plt.title(self.class_name(labels[i]))
            plt.axis('off')
            
        plt.tight_layout()
        plt.show()