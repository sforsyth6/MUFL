from binaryLoader import BinaryDataset, pad_collate_func, RandomChunkSampler

from torch.utils.data import Dataset, DataLoader

data = "/data/test/"

whole_dataset = BinaryDataset(data, sort_by_size=True, max_len=16000000)

train_loader = DataLoader(whole_dataset, batch_size=1, num_workers=1, collate_fn=pad_collate_func, 
                            sampler=RandomChunkSampler(whole_dataset,1))
