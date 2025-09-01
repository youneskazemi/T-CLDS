import numpy as np
import torch

import utils
import world


def BPR_train_original(dataset, recommend_model, loss_class, epoch):
    """
    One training epoch for BPR. Uses time-aware sampling if available:
      - utils.UniformSample_original returns: S, T, sam_time
        * S: np.array of shape (N, 3) with columns [user, pos, neg]
        * T: dict with 't_raw' (epoch seconds) and 't_model' (normalized float)
    """
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class

    allusers = list(range(dataset.n_users))
    S, T, sam_time = utils.UniformSample_original(
        allusers, dataset
    )  # S=[u,p,n], T={'t_raw','t_model'}

    users = torch.tensor(S[:, 0]).long()
    posItems = torch.tensor(S[:, 1]).long()
    negItems = torch.tensor(S[:, 2]).long()

    # Time tensors
    t_raw = torch.from_numpy(T["t_raw"]).float()  # epoch seconds (or numeric)
    t_model = torch.from_numpy(T["t_model"]).float()  # normalized for encoder

    # Move to device
    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    t_raw = t_raw.to(world.device)
    t_model = t_model.to(world.device)

    # Shuffle in unison
    users, posItems, negItems, t_raw, t_model = utils.shuffle(
        users, posItems, negItems, t_raw, t_model
    )

    total_batch = len(users) // world.config["bpr_batch_size"] + 1
    aver_loss = 0.0
    comp_sums = {"bpr": 0.0, "reg": 0.0, "attr": 0.0, "lbl": 0.0}

    for (
        batch_users,
        batch_pos,
        batch_neg,
        batch_t_raw,
        batch_t_model,
    ) in utils.minibatch(
        users,
        posItems,
        negItems,
        t_raw,
        t_model,
        batch_size=world.config["bpr_batch_size"],
    ):
        # stageOne now accepts optional time tensors
        total_loss, comps = bpr.stageOne(
            batch_users, batch_pos, batch_neg, epoch, batch_t_model, batch_t_raw
        )
        aver_loss += total_loss
        for k in comp_sums:
            comp_sums[k] += comps.get(k, 0.0)

    aver_loss = aver_loss / total_batch
    for k in comp_sums:
        comp_sums[k] = comp_sums[k] / total_batch
    return aver_loss, comp_sums


def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:  # [10, 20]
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret["precision"])
        recall.append(ret["recall"])
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
    return {
        "recall": np.array(recall),
        "precision": np.array(pre),
        "ndcg": np.array(ndcg),
    }


def Test(dataset, Recmodel, epoch, cold=False, w=None):
    u_batch_size = int(world.config["test_u_batch_size"])  # 100
    # dict
    if cold:
        testDict: dict = dataset.coldTestDict
    else:
        testDict: dict = dataset.testDict
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)  # 20
    results = {
        "precision": np.zeros(len(world.topks)),  # 2
        "recall": np.zeros(len(world.topks)),  # 2
        "ndcg": np.zeros(len(world.topks)),
        # Temporal metrics
        "tndcg": np.zeros(len(world.topks)),
        "trecall": np.zeros(len(world.topks)),
        "hr_time": np.zeros(len(world.topks)),
    }  # 2
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(
                f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}"
            )
        users_list = []
        rating_list = []
        groundTrue_list = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)

            rating[exclude_index, exclude_items] = -(1 << 10)
            _, rating_K = torch.topk(rating, k=max_K)
            del rating

            users_list.append(batch_users)

            rating_list.append(rating_K.cpu())

            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        pre_results = []
        for x in X:  # rating_list[i]   groundTrue_list[i]
            pre_results.append(test_one_batch(x))
        for result in pre_results:
            results["recall"] += result["recall"]
            results["precision"] += result["precision"]
            results["ndcg"] += result["ndcg"]
        results["recall"] /= float(len(users))
        results["precision"] /= float(len(users))
        results["ndcg"] /= float(len(users))

        # Calculate temporal metrics
        for k_idx, k in enumerate(world.topks):
            # Get predictions and ground truth for this k
            r = utils.getLabel(groundTrue_list, [rating[:k] for rating in rating_list])

            # Create user-item mapping for temporal metrics
            user_item_pairs = []
            for batch_idx, (batch_users, user_items) in enumerate(
                zip(users_list, groundTrue_list)
            ):
                for item in user_items:
                    # Map each item to its corresponding user in the batch
                    user_item_pairs.append((batch_users[0], item))

            # Temporal NDCG@K
            results["tndcg"][k_idx] = utils.temporal_NDCG_atK(
                groundTrue_list, r, k, dataset, user_item_pairs
            )

            # Temporal Recall@K
            results["trecall"][k_idx] = utils.temporal_Recall_atK(
                groundTrue_list, r, k, dataset, user_item_pairs
            )

            # Hit Ratio over Time (1 month window)
            results["hr_time"][k_idx] = utils.Hit_Ratio_over_Time(
                groundTrue_list,
                r,
                k,
                dataset,
                time_window_hours=24 * 30,
                user_item_pairs=user_item_pairs,
            )

        # Format output like the target
        print(results)
        print(
            f"[TEST] P@10 {results['precision'][0]:.4f} | R@10 {results['recall'][0]:.4f} | NDCG@10 {results['ndcg'][0]:.4f} || P@20 {results['precision'][1]:.4f} | R@20 {results['recall'][1]:.4f} | NDCG@20 {results['ndcg'][1]:.4f}"
        )
        print(
            f"[TEST] tNDCG@10 {results['tndcg'][0]:.4f} | tR@10 {results['trecall'][0]:.4f} | HR@10 {results['hr_time'][0]:.4f} || tNDCG@20 {results['tndcg'][1]:.4f} | tR@20 {results['trecall'][1]:.4f} | HR@20 {results['hr_time'][1]:.4f}"
        )
        print("[TEST] running evaluation...")
        return results
