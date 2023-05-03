from typing import List

from torch.utils.data import Subset
import numpy as np

from . import ClassificationDataset
from .smote import SMOTEDataset


def split_dataset_by_ids(dataset: ClassificationDataset, idss: List[List]) -> List[Subset]:
    if isinstance(dataset, SMOTEDataset):
        return split_dataset_by_ids_smote(dataset, idss)
    id_index_map = {id:index for (index, id) in enumerate(dataset.get_graph_identifiers())}

    subsets = list()
    for ids in idss:
        indices = [id_index_map[id] for id in ids]
        subsets.append(Subset(dataset, indices))
    return subsets


def split_dataset_by_ids_smote(dataset: ClassificationDataset, idss: List[List]) -> List[Subset]:
    id_index_map = {id:index for (index, id) in enumerate(dataset.get_graph_identifiers())}

    indicess = list()
    for ids in idss:
        # because of randomness, the interpolated graphs will not be found again
        # filter them and add in the end to first subset
        # assuming first subset is train set
        indices = [id_index_map[id] for id in ids if id in id_index_map]
        indicess.append(indices)
    rest = [i for i in range(len(dataset)) if all(i not in indices for indices in indicess)]
    indicess[0].extend(rest)
    return [Subset(dataset, indices) for indices in indicess]