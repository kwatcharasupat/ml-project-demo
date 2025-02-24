from omegaconf import DictConfig
import pytorch_lightning as pl
import torchaudio as ta

from .dataset import GTZANDataset


class GTZANDataModule(pl.LightningDataModule):
    @classmethod
    def from_dataset(
        cls, *, data_root: str, batch_size: int, num_workers: int, verbose: bool = True
    ):
        if verbose:
            print(f"Loading GTZAN from {data_root}")
            print("Loading training dataset")

        train_dataset = GTZANDataset(
            data_root=data_root,
            split="train",
        )

        if verbose:
            print("Loading validation dataset")

        val_dataset = GTZANDataset(
            data_root=data_root,
            split="val",
        )

        if verbose:
            print("Loading testing dataset")

        test_dataset = GTZANDataset(
            data_root=data_root,
            split="test",
        )

        return cls.from_datasets(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    @classmethod
    def from_config(cls, *, config: DictConfig):
        return cls.from_dataset(
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )


if __name__ == "__main__":
    gtzan = GTZANDataModule.from_dataset(
        data_root="/home/kwatchar3/Documents/data/gtzan",
        batch_size=32,
        num_workers=4,
    )

    train_dataloader = gtzan.train_dataloader()

    for i, batch in enumerate(train_dataloader):
        print(i, batch["audio"].shape, batch["label"])
