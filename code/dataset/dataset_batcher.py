import pathlib
import gzip

import numpy as np
import torch


class SlimDataset(torch.utils.data.Dataset):
    """ SlimDataset class for SLIM.
        SlimDataset class loads `*.pt.gz` data files. Each `*.pt.gz` file
        includes list of tuples, and these are rearanged to mini batches.

        Args:
        root_dir:       str
                        Path to root directory.

    """
    def __init__(
        self,
        root_dir: str,
    ) -> None:
        super().__init__()

        files = pathlib.Path(root_dir).expanduser().glob("*.pt.gz")
        self.record_list = sorted(files)

    def __len__(self) -> int:
        """Returns number of files and directories in root dir.

        Args: None

        Returns:
            len:    int
                    Number of objects in root dir.
        """

        return len(self.record_list)

    def __getitem__(self, index: int) -> list:
        """Loads data file and returns data with specified index.
        This method reads `<index>.pt.gz` file which includes a list of Tensors
        Args:
            index (int): Index number.
        Returns:
            datset: List of tensors
        """

        with gzip.open(self.record_list[index], "rb") as f:
            dataset = torch.load(f)

        return dataset
