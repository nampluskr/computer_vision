import os
import torch

from models.components.backbone import set_backbone_dir, get_backbone_dir
from dataloader import set_dataset_dir, get_dataset_dir, get_dataloaders
from registry import get_trainer, get_train_config


#####################################################################
# Global VARIABLES
#####################################################################

DATASET_DIR = "/mnt/d/datasets"
BACKBONE_DIR = "/mnt/d/backbones"
OUTPUT_DIR = "/mnt/d/outputs"
SEED = 42
NUM_WORKERS = 8
PIN_MEMORY = True
PERSISTENT_WORKERS = True


def set_globals(dataset_dir=None, backbone_dir=None, output_dir=None,
               seed=None, num_workers=None, pin_memory=None, persistent_workers=None,
               show_globals=False):
    global DATASET_DIR, BACKBONE_DIR, OUTPUT_DIR, SEED
    global NUM_WORKERS, PIN_MEMORY, PERSISTENT_WORKERS

    if dataset_dir is not None:
        DATASET_DIR = dataset_dir or get_dataset_dir()
    if backbone_dir is not None:
        BACKBONE_DIR = backbone_dir or get_backbone_dir()
    if output_dir is not None:
        OUTPUT_DIR = output_dir
    if seed is not None:
        SEED = seed

    if num_workers is not None:
        NUM_WORKERS = num_workers
    if pin_memory is not None:
        PIN_MEMORY = pin_memory
    if persistent_workers is not None:
        PERSISTENT_WORKERS = persistent_workers
        
    set_backbone_dir(BACKBONE_DIR)
    set_dataset_dir(DATASET_DIR)

    if show_globals:
        print_globals()


def get_globals():
    return {
        "dataset_dir": DATASET_DIR,
        "backbone_dir": BACKBONE_DIR,
        "output_dir": OUTPUT_DIR,
        "seed": SEED,
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
        "persistent_workers": PERSISTENT_WORKERS,
    }


def print_globals():
    config = get_globals()
    print("\n" + "="*70)
    print("Training Configuration")
    print("-"*70)
    for key, value in config.items():
        print(f"  {key:20s}: {value}")
    print("-"*70 )


def set_seed(seed=42):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def count_parameters(trainer):
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print()
    print(f" > Total params.:         {total_params:,}")
    print(f" > Trainable params.:     {trainable_params:,}")

    if trainer.optimizer is not None:
        optim_params = sum(p.numel() for group in trainer.optimizer.param_groups for p in group['params'])
        print(f" > Optimizer params.:     {optim_params:,}")


def train(dataset_type, category, model_type, num_epochs=None, batch_size=None, img_size=None, normalize=None):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    try:
        train_config = get_train_config(model_type)
        num_epochs = num_epochs or train_config["num_epochs"]
        img_size = img_size or train_config["img_size"]
        batch_size = batch_size or train_config["batch_size"]
        normalize = normalize if normalize is not None else train_config["normalize"]

        if isinstance(dataset_type, list):
            dtype_str = "-".join(dataset_type)
        else:
            dtype_str = dataset_type
        
        if isinstance(category, list):
            cat_str = "-".join(category)
        elif category == "all":
            cat_str = "all"
        else:
            cat_str = category

        print(f"\n{'='*70}")
        print(f"Training: {model_type} | {dtype_str}/{cat_str}")
        print(f"{'-'*70}")
        print(f"  Max. Epochs: {num_epochs}")
        print(f"  Batch Size:  {batch_size}")
        print(f"  Image Size:  {img_size}")
        print(f"  Normalize:   {normalize}")
        print(f"{'='*70}\n")

        result_dir = os.path.join(OUTPUT_DIR, dtype_str, cat_str, model_type)
        desc = f"{dtype_str}_{cat_str}_{model_type}"
        weight_path = os.path.join(result_dir, f"model_{desc}_epochs-{num_epochs}.pth")

        set_seed(seed=SEED)

        train_loader, test_loader = get_dataloaders(
            dataset_dir=DATASET_DIR,
            dataset_type=dataset_type,
            category=category,
            img_size=img_size,
            batch_size=batch_size,
            normalize=normalize,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=PERSISTENT_WORKERS
        )

        trainer = get_trainer(model_type, img_size=img_size)
        count_parameters(trainer)
        print_memory("After Model Creation")

        trainer.fit(train_loader, num_epochs, valid_loader=test_loader, weight_path=weight_path)
        print_memory("After Training")

        trainer.save_maps(test_loader, result_dir=result_dir, desc=desc, normalize=normalize,
            skip_normal=True)
        trainer.save_histogram(test_loader, result_dir=result_dir, desc=desc)
        trainer.save_results(test_loader, result_dir=result_dir, desc=desc)

    except Exception as e:
        print(f"\n{'!'*70}")
        print(f"ERROR in {model_type}: {e}")
        print(f"{'!'*70}\n")
        import traceback
        traceback.print_exc()

    finally:
        if 'trainer' in locals():
            del trainer
        if 'train_loader' in locals():
            clear_dataloader(train_loader)
            del train_loader
        if 'test_loader' in locals():
            clear_dataloader(test_loader)
            del test_loader

        clear_memory(print_summary=True)


def clear_dataloader(dataloader):
    try:
        if hasattr(dataloader, '_iterator') and dataloader._iterator is not None:
            del dataloader._iterator
        if hasattr(dataloader, 'worker_pids') and len(dataloader.worker_pids) > 0:
            dataloader._shutdown_workers()
        if hasattr(dataloader, '_persistent_workers') and dataloader._persistent_workers:
            dataloader._persistent_workers = False
    except Exception as e:
        print(f" > [Warning] Dataloader cleanup failed: {e}")


def clear_memory(print_summary=True, stage="After Cleanup"):
    import gc

    collected = gc.collect()
    if print_summary:
        print(f" > Python GC: Collected {collected} objects")

    if torch.cuda.is_available():
        if print_summary:
            before_allocated = torch.cuda.memory_allocated() / 1024**3
            before_reserved = torch.cuda.memory_reserved() / 1024**3

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        if print_summary:
            after_allocated = torch.cuda.memory_allocated() / 1024**3
            after_reserved = torch.cuda.memory_reserved() / 1024**3
            freed_allocated = before_allocated - after_allocated
            freed_reserved = before_reserved - after_reserved

            if freed_allocated > 0.01 or freed_reserved > 0.01:  # 10MB 이상만 표시
                print(f" > Freed: {freed_allocated:.2f} GB allocated, {freed_reserved:.2f} GB reserved")

            print_memory(stage)

        torch.cuda.reset_peak_memory_stats()


def print_memory(stage=""):
    if not torch.cuda.is_available():
        return

    device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
    free_memory = total_memory - allocated
    usage = allocated / total_memory * 100

    stage_str = f"[{stage}]" if stage else ""
    print(f"\n{'GPU Memory ' + stage_str:-^70}")
    print(f"  Allocated: {allocated:.2f}GB ({usage:.1f}%) | Reserved: {reserved:.2f}GB | Peak: {max_allocated:.2f}GB")
    print(f"  Free: {free_memory:.2f}GB / Total: {total_memory:.2f}GB")
    print(f"{'-'*70}")


def train_models(dataset_type, categories, models, clear_memory_between=True):
    num_categories = len(categories)
    num_models = len(models)
    total_runs = num_categories * num_models

    print(f"\n{'='*70}")
    print(f"  Training {num_models} model(s) on {num_categories} category(ies)")
    print(f"  Total runs: {total_runs}")
    print(f"{'='*70}\n")

    all_results = []
    run_count = 0
    for cat_idx, category in enumerate(categories, 1):
        print(f"\n{'#'*70}")
        print(f"# Category [{cat_idx}/{num_categories}]: {category}")
        print(f"{'#'*70}\n")

        category_results = []
        for model_type in models:
            run_count += 1
            print(f"{'='*70}")
            print(f"  Run [{run_count}/{total_runs}]: {model_type} | {dataset_type}/{category}")
            print(f"{'='*70}")

            try:
                train(dataset_type, category, model_type)
                result = {
                    "dataset": dataset_type,
                    "category": category,
                    "model": model_type,
                    "status": "success"
                }
                category_results.append(result)
                all_results.append(result)

                if clear_memory_between and run_count < total_runs:
                    print("\n > Additional memory cleanup between runs...\n")
                    clear_memory(print_summary=False)
                    import time
                    time.sleep(1)

            except Exception as e:
                print(f"\n{'!'*70}")
                print(f"[ERROR] Failed to train {model_type}: {e}")
                print(f"{'!'*70}\n")
                import traceback
                traceback.print_exc()

                result = {
                    "dataset": dataset_type,
                    "category": category,
                    "model": model_type,
                    "status": "failed",
                    "error": str(e)
                }
                category_results.append(result)
                all_results.append(result)

        # Print category summary
        cat_success = sum(1 for r in category_results if r["status"] == "success")
        cat_failed = sum(1 for r in category_results if r["status"] == "failed")

        print(f"\n{f'Category Summary: {category}':-^70}")
        print(f"  Models:          {len(category_results)}")
        print(f"  Successful:      {cat_success}")
        print(f"  Failed:          {cat_failed}")
        print(f"{'='*70}\n")

    # Print overall summary
    print(f"\n{'='*70}")
    print(f"{'OVERALL TRAINING SUMMARY':^70}")
    print(f"{'='*70}")

    success_count = sum(1 for r in all_results if r["status"] == "success")
    failed_count = sum(1 for r in all_results if r["status"] == "failed")

    print(f"  Total Runs:      {total_runs}")
    print(f"  Successful:      {success_count}")
    print(f"  Failed:          {failed_count}")
    print(f"  Success Rate:    {success_count/total_runs*100:.1f}%")

    if failed_count > 0:
        print(f"\n  Failed Runs:")
        for result in all_results:
            if result["status"] == "failed":
                error_msg = result.get('error', 'Unknown error')
                print(f"    - {result['dataset']}/{result['category']}/{result['model']}: {error_msg}")

    print(f"{'='*70}\n")
    return all_results


if __name__ == "__main__":
    
    #################################################################
    # Open datasets: MVTec / VisA / BTAD
    #################################################################

    set_globals(
        dataset_dir="/mnt/d/datasets",
        backbone_dir="/mnt/d/backbones",
        output_dir="/mnt/d/outputs",
        seed=42,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        show_globals=True
    )

    train_models(dataset_type="mvtec",
        categories=["wood", "grid", "tile"],
        models=["fastflow", "stfpm"]
    )

    train("mvtec", "wood", "stfpm", num_epochs=20)
    train("visa", "macaroni1", "stfpm", num_epochs=20)
    train("btad", "03", "stfpm", num_epochs=20)

    #################################################################    
    # Custon Dataset
    #################################################################
    
    set_globals(
        dataset_dir="/mnt/d/datasets/custom",
        show_globals=True,
    )
    
    train(["module1"], "tile", "stfpm", num_epochs=20)
    train(["module1"], ["grid", "tile"], "stfpm", num_epochs=20)