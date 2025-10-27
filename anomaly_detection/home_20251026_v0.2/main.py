""" filename: main.py """
import os
os.environ['DATA_DIR'] = '/home/namu/myspace/NAMU/datasets' 
os.environ['BACKBONE_DIR'] = '/home/namu/myspace/NAMU/backbones'

from trainers import set_seed, set_logger, get_trainer
from transforms import get_train_transform, get_test_transform, get_mask_transform
from dataloaders import get_train_loader, get_test_loader


def main():
    dataset_name = 'btad'
    category = '01'
    model_name = 'supersimplenet-supervised'
    img_size = 256
    batch_size = 32
    normalize = True
    num_epochs = 20
    device = 'cuda'
    seed = 42

    output_dir = './outputs'
    run_name = f"{model_name}_{dataset_name}_{category}"

    set_seed(seed)
    logger = set_logger(output_dir, run_name)

    logger.info("="*70)
    logger.info("Training Configuration")
    logger.info("="*70)
    logger.info(f"Model: {model_name}")
    logger.info(f"Dataset: {dataset_name}/{category}")
    logger.info(f"Image size: {img_size}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Epochs: {num_epochs}")
    logger.info(f"Device: {device}")
    logger.info("="*70)
    
    # Create transforms
    train_transform = get_train_transform(img_size=img_size, normalize=normalize)
    test_transform = get_test_transform(img_size=img_size, normalize=normalize)
    mask_transform = get_mask_transform(img_size=img_size)
    
    # Create dataloaders
    train_loader = get_train_loader(
        dataset_name=dataset_name,
        category=category,
        transform=train_transform,
        batch_size=batch_size,
    )
    
    valid_loader = get_test_loader(
        dataset_name=dataset_name, 
        category=category,
        transform=test_transform,
        mask_transform=mask_transform,
        batch_size=batch_size,
    )
    
    logger.info(f"Train dataset size: {len(train_loader.dataset)}, batch_size: {train_loader.batch_size}")
    logger.info(f"Valid dataset size: {len(valid_loader.dataset)}, batch_size: {valid_loader.batch_size}")
    
    trainer = get_trainer(model_name=model_name, logger=logger)
    logger.info("Trainer created")
    logger.info(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    
    trainer.fit(train_loader, num_epochs, valid_loader=valid_loader)

if __name__ == "__main__":
    main()
