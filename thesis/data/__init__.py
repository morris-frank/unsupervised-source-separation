from torch.utils import data


class Dataset(data.Dataset):
    def loader(self, batch_size: int) -> data.DataLoader:
        return data.DataLoader(self, batch_size=batch_size, num_workers=1,
                               shuffle=True)

    @property
    def n_sources(self):
        _, S = self[0]
        return S.shape[0]

    def __str__(self) -> str:
        return f'{type(self).__name__} with <{len(self):>7} signals>'
