from __future__ import annotations

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from curvature_bench.registry import DATA_REGISTRY


def _make_image_transform(cfg: dict):
    normalize = cfg.get("normalize", True)

    transform_list = [transforms.ToTensor()]

    if normalize:
        mean = tuple(cfg.get("mean", [0.5]))
        std = tuple(cfg.get("std", [0.5]))
        transform_list.append(transforms.Normalize(mean, std))

    return transforms.Compose(transform_list)


def _make_loaders_from_dataset_cls(dataset_cls, cfg: dict):
    root = cfg.get("root", "./outputs/data")
    batch_size = int(cfg.get("batch_size", 128))
    num_workers = int(cfg.get("num_workers", 2))
    pin_memory = bool(cfg.get("pin_memory", True))

    transform = _make_image_transform(cfg)

    train_set = dataset_cls(root=root, train=True, download=True, transform=transform)
    val_set = dataset_cls(root=root, train=False, download=True, transform=transform)

    train_subset = cfg.get("train_subset", None)
    val_subset = cfg.get("val_subset", None)

    if train_subset is not None:
        train_set = Subset(train_set, range(int(train_subset)))

    if val_subset is not None:
        val_set = Subset(val_set, range(int(val_subset)))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


@DATA_REGISTRY.register("fashion_mnist")
def build_fashion_mnist(cfg: dict):
    return _make_loaders_from_dataset_cls(datasets.FashionMNIST, cfg)


@DATA_REGISTRY.register("mnist")
def build_mnist(cfg: dict):
    return _make_loaders_from_dataset_cls(datasets.MNIST, cfg)


@DATA_REGISTRY.register("cifar10")
def build_cifar10(cfg: dict):
    return _make_loaders_from_dataset_cls(datasets.CIFAR10, cfg)


def build_dataloaders(cfg: dict):
    return DATA_REGISTRY.build(cfg)