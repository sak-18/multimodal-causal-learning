# crisis_mmd_dataset.py

import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import Counter, OrderedDict
from nltk.tokenize import word_tokenize
from torchvision.datasets.folder import pil_loader

class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order of elements encountered.
    
    This class combines the functionality of `Counter` and `OrderedDict`:
    - Counts occurrences like a `Counter`.
    - Maintains insertion order like an `OrderedDict`.

    Example:
        oc = OrderedCounter()
        oc.update(['apple', 'banana', 'apple'])
        print(oc)  # Output: OrderedCounter({'apple': 2, 'banana': 1})

    """
    def __repr__(self):
        """Return a string representation of the `OrderedCounter`."""
        return "%s(%r)" % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        """Helper for pickle support."""
        return self.__class__, (OrderedDict(self),)


class CrisisMMDataset(Dataset):
    """CrisisMMDataset class to handle image-text pairs with vocabulary management.

    Attributes:
        vocab_size (int): Size of the vocabulary.
        max_sequence_length (int): Maximum sequence length for padding.
        transform (callable): Transformations to apply to images.
        vocab_filepath (str): Path to saved vocabulary, if exists.
    """

    def __init__(self, annotation_file, image_folder, mode="train", task="informative", transform=None, vocab_filepath="vocab.json", max_sequence_length=32):
        """
        Args:
            annotation_file (str): Path to the task-specific TSV file.
            image_folder (str): Path to the directory containing images.
            mode (str): One of 'train', 'val', or 'test'.
            task (str): The specific task, either 'informative', 'humanitarian', or 'severity'.
            transform (callable): Optional transform to be applied to images.
            vocab_filepath (str): Optional path to a saved vocabulary.
            max_sequence_length (int): Fixed sequence length for text padding.
        """
        self.mode = mode
        self.task = task
        self.image_folder = image_folder
        self.transform = transform
        self.max_sequence_length = max_sequence_length

        # Load annotations
        if not os.path.exists(annotation_file):
            raise FileNotFoundError(f"Annotation file {annotation_file} not found.")
        self.data = pd.read_csv(annotation_file, sep='\t')

        # Map task names to specific label columns
        self.task_label_columns = {
            "informative": "label_text_image",
            "humanitarian": "label_text_image",
            "damage": "label"
        }
        self.text_label_columns = {
            "informative": "label_text",
            "humanitarian": "label_text"
        }
        self.image_label_columns = {
            "informative": "label_image",
            "humanitarian": "label_image"
        }

        # Load or create vocabulary
        self.vocab_filepath = vocab_filepath
        self.w2i, self.i2w = self._load_or_create_vocab(vocab_filepath)
        self.vocab_size = len(self.w2i)

    def _load_or_create_vocab(self, vocab_filepath):
        """Load or create vocabulary from the training data."""
        if vocab_filepath and os.path.exists(vocab_filepath):
            # Load pre-existing vocabulary
            with open(vocab_filepath, 'r') as f:
                vocab = json.load(f)
            w2i, i2w = vocab['w2i'], vocab['i2w']
        else:
            # Create vocabulary from training data
            if self.mode != "train":
                raise ValueError("Vocabulary should be created from training data.")
            w2i, i2w = self._create_vocab(vocab_filepath)
        return w2i, i2w

    def __len__(self):
        return len(self.data)  # Return the number of samples in the dataset

    def _create_vocab(self, vocab_filepath):
        """Create and save vocabulary from training data."""
        print(f"Creating vocabulary as '{vocab_filepath}'...")

        vocab_counter = OrderedCounter()
        if self.data.empty:
            raise ValueError("The training data is empty, unable to create a vocabulary.")

        for text in self.data['tweet_text'].dropna():
            tokens = word_tokenize(text.lower())
            vocab_counter.update(tokens)

        # Initialize vocabulary with special tokens
        w2i = {"{pad}": 0, "{eos}": 1, "{unk}": 2}
        i2w = {0: "{pad}", 1: "{eos}", 2: "{unk}"}

        # Add words meeting minimum frequency requirement to vocabulary
        for idx, (word, _) in enumerate(vocab_counter.most_common(), start=len(w2i)):
            w2i[word] = idx
            i2w[idx] = word

        # Save vocabulary
        vocab = {'w2i': w2i, 'i2w': i2w}
        if vocab_filepath:
            with open(vocab_filepath, 'w') as f:
                json.dump(vocab, f)
            print(f"Vocabulary saved to {vocab_filepath} with {len(w2i)} entries.")
        return w2i, i2w

    def _tokenize_text(self, text):
        """Tokenize and pad text."""
        tokens = word_tokenize(text.lower()) if isinstance(text, str) else []
        indices = [self.w2i.get(token, self.w2i["{unk}"]) for token in tokens]
        # Pad sequence to max_sequence_length
        indices = indices[:self.max_sequence_length - 1] + [self.w2i["{eos}"]]
        padded_indices = torch.full((self.max_sequence_length,), self.w2i["{pad}"], dtype=torch.long)
        padded_indices[:len(indices)] = torch.tensor(indices, dtype=torch.long)
        return padded_indices

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row.get('tweet_text', "")
        image_path = os.path.join(self.image_folder, row['image'])

        # Load and transform image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file {image_path} not found.")
        image = pil_loader(image_path)
        if self.transform:
            image = self.transform(image)

        # Tokenize text and pad to max_sequence_length
        text_indices = self._tokenize_text(text)
        text_indices_onehot = torch.nn.functional.one_hot(text_indices.clone().detach(), num_classes=self.vocab_size).float()

        # Create a mapping from string labels to numerical indices
        label_column = self.task_label_columns.get(self.task)
        label_str = row[label_column] if pd.notna(row[label_column]) else None

        # Create a dictionary to map string labels to numerical values
        label_mapping = {label: i for i, label in enumerate(self.data[label_column].unique())}

        # Convert string labels to numerical labels
        label = torch.tensor(label_mapping.get(label_str, -1), dtype=torch.long)

        # Select modality-specific labels if available
        text_label_str = row.get(self.text_label_columns.get(self.task), None)
        image_label_str = row.get(self.image_label_columns.get(self.task), None)

        text_label = torch.tensor(label_mapping.get(text_label_str, -1), dtype=torch.long)
        image_label = torch.tensor(label_mapping.get(image_label_str, -1), dtype=torch.long)

        # If needed, apply one-hot encoding to labels (for multi-class classification)
        num_classes = len(label_mapping)
        if label >= 0:  # Only apply one-hot encoding if the label is valid
            label = torch.nn.functional.one_hot(label, num_classes=num_classes).float()
        if text_label >= 0:
            text_label = torch.nn.functional.one_hot(text_label, num_classes=num_classes).float()
        if image_label >= 0:
            image_label = torch.nn.functional.one_hot(image_label, num_classes=num_classes).float()

        if self.task == "damage":
            return {
                "text": text_indices_onehot,  # One-hot encoded text
                "image": image,
                "label": label,            # Overall task label (one-hot encoded)
            }


        return {
            "text": text_indices_onehot,  # One-hot encoded text
            "image": image,
            "label": label,            # Overall task label (one-hot encoded)
            "text_label": text_label,  # Task-specific text label (one-hot encoded)
            "image_label": image_label  # Task-specific image label (one-hot encoded)
        }



    def __len__(self):
        return len(self.data)
