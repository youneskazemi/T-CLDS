import os
from time import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import optim

import world


class BPRLoss:
    def __init__(self, recmodel, config):
        self.model = recmodel
        self.weight_decay = config["decay"]  # 1e-4
        self.lr = config["lr"]
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)

    def stageOne(self, users, pos, neg, epoch, t_model=None, t_raw=None):
        """
        Single optimization step.
        If the model supports time-aware loss, it will use t_model/t_raw.
        """
        # loss components from model
        loss_bpr, reg_loss, attr_loss, lbl_loss, tag = self.model.bpr_loss(
            users, pos, neg, epoch, t_model=t_model, t_raw=t_raw
        )
        # scale components according to original recipe
        attr_scaled = attr_loss * 9
        reg_scaled = reg_loss * self.weight_decay
        total_loss = loss_bpr + reg_scaled + attr_scaled
        if tag == 1:
            total_loss = total_loss + lbl_loss

        self.opt.zero_grad()
        total_loss.backward()
        self.opt.step()

        components = {
            "bpr": float(loss_bpr.detach().cpu().item()),
            "reg": float(reg_scaled.detach().cpu().item()),
            "attr": float(attr_scaled.detach().cpu().item()),
            "lbl": float(lbl_loss.detach().cpu().item()) if tag == 1 else 0.0,
        }
        return float(total_loss.detach().cpu().item()), components


def UniformSample_original(users, dataset):
    """
    Original sampler, extended to also return timestamp arrays:
      returns S, T, [timings]
        - S: np.array of shape (N,3) with [user, pos, neg]
        - T: dict with
             * 't_raw'   : np.array of shape (N,) epoch seconds (float32)
             * 't_model' : np.array of shape (N,) normalized time for encoder (float32)
    """
    total_start = time()
    user_num = dataset.trainDataSize

    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    S = []
    t_raw = []
    t_model = []
    sample_time1 = 0.0
    sample_time2 = 0.0
    for i, user in enumerate(users):
        start = time()
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        # negative
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])

        # time for positive edge (user, positem)
        tr, tm = dataset.get_ui_time(user, positem)
        t_raw.append(tr)
        t_model.append(tm)

        end = time()
        sample_time1 += end - start

    total = time() - total_start
    T = {
        "t_raw": np.asarray(t_raw, dtype=np.float32),
        "t_model": np.asarray(t_model, dtype=np.float32),
    }
    return np.array(S), T, [total, sample_time1, sample_time2]


# ===================end samplers==========================
# =====================utils====================================


def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def getFileName():
    if world.model_name == "bpr":
        file = f"mf-{world.dataset}-{world.config['latent_dim_rec']}.pth.tar"
    elif world.model_name in ["LightGCN", "CLDS"]:
        file = (
            f"{world.model_name}-{world.dataset}-{world.config['layer']}layer-"
            f"{world.config['latent_dim_rec']}.pth.tar"
        )
    return os.path.join(world.FILE_PATH, file)


def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get("batch_size", world.config["bpr_batch_size"])

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i : i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i : i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):
    require_indices = kwargs.get("indices", False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError("All inputs to shuffle must have " "the same length.")

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


# ====================Metrics==============================
# =========================================================
def RecallPrecision_ATk(test_data, r, k):
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n
    return {"recall": recall, "precision": precis}


def MRRatK_r(r, k):
    pred_data = r[:, :k]
    scores = np.log2(1.0 / np.arange(1, k + 1))
    pred_data = pred_data / scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)


def NDCGatK_r(test_data, r, k):
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1.0 / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1.0 / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.0] = 1.0
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.0
    return np.sum(ndcg)


def AUC(all_item_scores, dataset, test_data):
    r_all = np.zeros((dataset.m_items,))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype("float")


# ====================end Metrics=============================
# =========================================================
