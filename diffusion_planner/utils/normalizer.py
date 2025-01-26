import torch


class StateNormalizer:
    def __init__(self, mean, std):
        self.mean = torch.as_tensor(mean)
        self.std = torch.as_tensor(std)

    def __call__(self, data):
        return (data - self.mean.to(data.device)) / self.std.to(data.device)

    def inverse(self, data):
        return data * self.std.to(data.device) + self.mean.to(data.device)

    def to_dict(self):
        return {
            "mean": self.mean.detach().cpu().numpy().tolist(),
            "std": self.std.detach().cpu().numpy().tolist()
        }


class ObservationNormalizer:
    def __init__(self, normalization_dict):
        self._normalization_dict = normalization_dict

    def __call__(self, data):
        for k, v in self._normalization_dict.items():
            if k not in data:  # Check if key `k` exists in `data`
                continue
            mask = torch.sum(torch.ne(data[k], 0), dim=-1) == 0
            data[k] = (data[k] - v["mean"].to(data[k].device)) / v["std"].to(data[k].device)
            data[k][mask] = 0
        return data

    def inverse(self, data):
        for k, v in self._normalization_dict.items():
            if k not in data:  # Check if key `k` exists in `data`
                continue
            mask = torch.sum(torch.ne(data[k], 0), dim=-1) == 0
            data[k] = data[k] * v["std"].to(data[k].device) + v["mean"].to(data[k].device)
            data[k][mask] = 0
        return data

    def to_dict(self):
        return {k: {kk: vv.detach().cpu().numpy().tolist() for kk, vv in v.items()} for k, v in self._normalization_dict.items()}