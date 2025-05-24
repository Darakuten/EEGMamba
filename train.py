from tqdm import tqdm
import os, torch, random
import wandb
import numpy as np
from omegaconf import open_dict
from dataclass.god import GODDatasetBase, GODCollator
from utils.get_dataloaders import get_samplers
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader
from eegMamba import EEGMamba
from model import Classifier
from debug_utils import (enable_debug, disable_debug, debug_print, 
                        check_gradients, debug_step, debug_summary, timing)

device = "cuda" if torch.cuda.is_available() else "cpu"

def topk_accuracy(logits, labels, k=5):

    topk_preds = torch.topk(logits, k, dim=-1).indices  # shape: (B, k)

    matches = (topk_preds == labels.unsqueeze(-1))  # shape: (B, k)

    topk_correct = matches.any(dim=-1)  # shape: (B,)

    accuracy = topk_correct.float().mean().item()
    return accuracy

@timing("training_run")
def run(args):
    # Enable debugging
    debug_mode = getattr(args, 'debug_mode', True)
    save_debug_tensors = getattr(args, 'save_debug_tensors', False)
    
    if debug_mode:
        enable_debug(save_tensors=save_debug_tensors)
        debug_print("=== EEGMamba Training Started ===", 'blue')
        debug_print(f"Device: {device}", 'cyan')
    else:
        disable_debug()

    from utils.reproducibility import seed_worker
    # NOTE: We do need it (IMHO).
    if args.reproducible:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        g = torch.Generator()
        g.manual_seed(0)
        seed_worker = seed_worker
    else:
        g = None
        seed_worker = None

    source_dataset = GODDatasetBase(args, 'train')

    # train_dataset, val_dataset = torch.utils.data.random_split(source_dataset, [train_size, val_size])
    ind_tr = list(range(0, 3000)) + list(range(3600, 6600)) #+ list(range(7200, 21600)) # + list(range(7200, 13200)) + list(range(14400, 20400))
    ind_te = list(range(3000,3600)) + list(range(6600, 7200)) # + list(range(13200, 14400)) + list(range(20400, 21600))
    train_dataset = Subset(source_dataset, ind_tr)
    val_dataset   = Subset(source_dataset, ind_te)

    with open_dict(args):
        args.num_subjects = source_dataset.num_subjects
        print('num subject is {}'.format(args.num_subjects))


    if args.use_sampler:
        test_size = 50# 重複サンプルが存在するのでval_dataset.Y.shape[0]
        train_loader, test_loader = get_samplers(
            train_dataset,
            val_dataset,
            args,
            test_bsz=test_size,
            collate_fn=GODCollator(args),)

    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size= args.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
        test_loader = DataLoader(
            val_dataset,
            batch_size=50, # args.batch_size,
            drop_last=True,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

    if args.use_wandb:
        wandb.config = {k: v for k, v in dict(args).items() if k not in ["root_dir", "wandb"]}
        wandb.init(
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=wandb.config,
            save_code=True,
        )
        wandb.run.name = args.wandb.run_name + "_" + args.split_mode
        wandb.run.save()

    model = EEGMamba(args).to(device)
    classifier = Classifier(args)

    loss_func = torch.nn.MSELoss(reduction="mean")


    optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr,
        )

    best_acc = 0
    pbar = tqdm(range(args.epochs))
    for epoch in pbar:
        pbar.set_description("training {}/{} epoch".format(epoch, args.epochs))
        debug_print(f"=== Epoch {epoch}/{args.epochs} ===", 'magenta')

        model.train()
        pbar2 = tqdm(train_loader)
        train_losses = []
        trainTop1accs = []
        trainTop10accs = []
        for i, batch in enumerate(pbar2):
            if len(batch) == 3:
                x, labels, subject_idxs = batch
            elif len(batch) == 4:
                x, labels, subject_idxs, chunkIDs = batch
                assert (
                    len(chunkIDs.unique()) == x.shape[0]
                ), "Duplicate segments in batch are not allowed. Aborting."
            else:
                raise ValueError("Unexpected number of items from dataloader.")

            x, labels = x.to(device), labels.to(device)
            # print("LABEL: ", labels[0])
            # print("\n\n--------------------------\n\n", x.shape)
            logits = model(x)
            loss = loss_func(logits, labels)

            train_losses.append(loss.item())

            with torch.no_grad():
              Top1acc, Top10acc = classifier(logits, labels)  # ( 250, 1024, 360 )

            trainTop1accs.append(Top1acc)
            trainTop10accs.append(Top10acc)
            if i % args.log_step == 0:
                print(f"Current Loss: {loss.item()}")
                print(f"Current Top1Acc: {Top1acc}")
                print(f"Current Top10Acc: {Top10acc}")

            optimizer.zero_grad()
            loss.backward()
            
            # Check gradients every 10 steps
            if debug_mode and i % 10 == 0:
                grad_norms = check_gradients(model)
            
            optimizer.step()
            debug_step()

        model.eval()
        pbar3 = tqdm(test_loader)
        test_losses = []
        testTop1accs = []
        testTop10accs = []
        for batch in pbar3:
            with torch.no_grad():
                if len(batch) == 3:
                    x, labels, subject_idxs = batch
                elif len(batch) == 4:
                    x, labels, subject_idxs, chunkIDs = batch
                    assert (
                        len(chunkIDs.unique()) == x.shape[0]
                    ), "Duplicate segments in batch are not allowed. Aborting."
                else:
                    raise ValueError("Unexpected number of items from dataloader.")

                x, labels = x.to(device), labels.to(device)

                logits = model(x)

                loss = loss_func(logits, labels)

                test_losses.append(loss.item())

                testTop1acc, testTop10acc = classifier(logits, labels, test=True)  # ( 250, 1024, 360 )

            testTop1accs.append(testTop1acc)
            testTop10accs.append(testTop10acc)

        print(f"Current Loss: {loss.item()}")
        print(f"Current Top1Acc: {testTop1acc}")
        print(f"Current Top10Acc: {testTop10acc}")

        if args.use_wandb:
            performance_now = {
                "epoch": epoch,
                "train_loss": np.mean(train_losses),
                "test_loss": np.mean(test_losses),
                "trainTop1acc": np.mean(trainTop1accs),
                "trainTop10acc": np.mean(trainTop10accs),
                "testTop1acc": np.mean(testTop1accs),
                "testTop10acc": np.mean(testTop10accs),
                "lrate": optimizer.param_groups[0]["lr"],
                "temp": 0,
            }
            wandb.log(performance_now)

        savedir = os.path.join(args.save_root, 'weights')
        last_weight_file = os.path.join(savedir, "model_last.pt")
        torch.save(model.state_dict(), last_weight_file)
        print('model is saved as ', last_weight_file)
        if best_acc < np.mean(testTop10accs):
            best_weight_file = os.path.join(savedir, "model_best.pt")
            torch.save(model.state_dict(), best_weight_file)
            best_acc =  np.mean(testTop10accs)
            print('best model is updated !!, {}'.format(best_acc), best_weight_file)
            debug_print(f"New best accuracy: {best_acc:.4f}", 'green')
    
    # Print debug summary at the end
    if debug_mode:
        debug_summary()

if __name__ == "__main__":
    from hydra import initialize, compose
    with initialize(version_base=None, config_path="configs"):
        args = compose(config_name='eegmamba')
    if not os.path.exists(os.path.join(args.save_root, 'weights')):
        os.makedirs(os.path.join(args.save_root, 'weights'))
    run(args)