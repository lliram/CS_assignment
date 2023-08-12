import numpy as np
import torch
from torch import nn

from definitions import TaskType

img_size = 512
aud_size = 192

class MatchNet(nn.Module):
    """
    Network for voice-face matching
    """
    def __init__(self):
        super().__init__()

        self.img_layer = nn.Linear(img_size, aud_size)
        self.aud_layer = nn.Linear(aud_size, aud_size)

        self.merged_stack = nn.Sequential(
            nn.Linear(aud_size * 2, aud_size),
            nn.ReLU(),
            nn.Linear(aud_size, 2),
        )

    def forward(self, x):
        x_aud, x_img = torch.split(x, [aud_size, img_size], dim=1)
        x_img = self.img_layer(x_img)
        x_aud = self.aud_layer(x_aud)
        x = torch.cat((x_aud, x_img), dim=1)
        x = self.merged_stack(x)
        return x


class VFFNet(nn.Module):
    """
    Network for voice-face-face arbitration
    """
    def __init__(self):
        super().__init__()

        self.img_layer = nn.Linear(img_size, aud_size)
        self.aud_layer = nn.Linear(aud_size, aud_size)

        self.merged_stack = nn.Sequential(
            nn.Linear(aud_size * 3, 192),
            nn.ReLU(),

            nn.Linear(192, 2),

        )

    def forward(self, x):
        x_aud, x_img_left, x_img_right = torch.split(x, [aud_size, img_size, img_size], dim=1)
        #from symmetry, both image should use the same layer
        x_img_left = self.img_layer(x_img_left)
        x_img_right = self.img_layer(x_img_right)
        x_aud = self.aud_layer(x_aud)
        x = torch.cat((x_aud, x_img_left, x_img_right), dim=1)
        x = self.merged_stack(x)
        return x


class NNModel:
    """
    Class for training and inference of the networks
    """
    def __init__(self, task_type: TaskType):
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Using {self.device} device")
        if task_type is TaskType.VF_MATCHING:
            self.model = MatchNet()
        elif task_type is TaskType.VFF_ARBITRATION:
            self.model = VFFNet()
        else:
            raise Exception('..')

        self.model.to(self.device)
        print(self.model)

        self.x_mean = None
        self.x_std = None

    def train(self, X_train, y_train, X_val, y_val,
              lr, batch_size, num_epochs,
              scale_data=True):

        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        if scale_data:
            # Shift training data to zero mean and scale it (z-scaling)
            # validation set is using the training data mean and std
            x_mean, x_std = X_train.mean(axis=0), X_train.std(axis=0)
            X_train = (X_train - x_mean) / x_std
            X_val = (X_val - x_mean) / x_std
            self.x_mean, self.x_std= x_mean, x_std
        else:
            self.x_mean, self.x_std = 0, 1


        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.int64)

        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.int64)

        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)


        N = len(y_train)
        for epoch in range(num_epochs):
            for batch_num, i in enumerate(range(0, N, batch_size)):
                X_batch = X_train[i: i+batch_size]
                y_batch = y_train[i: i+batch_size]
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                pred = self.model(X_batch)
                loss = loss_fn(pred, y_batch)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()


            val_pred = self.model(X_val)
            val_loss = loss_fn(val_pred, y_val)
            val_acc = self._acc(val_pred, y_val)
            print(f'Epoch {epoch+1 } / {num_epochs}, , loss = {loss.item()}, validation loss = {val_loss.item()}, '
                  f'validation accuracy = {val_acc}, ')


    def score(self, X, y, scale_data=True):
        """
        Compute accuracy score
        """
        y_hat = self.predict(X, scale_data)
        acc = (y == y_hat).mean()
        return acc

    def predict(self, X, scale_data=True):
        p = self.predict_prob(X, scale_data)
        return np.argmax(p, axis=1)

    def predict_prob(self, X, scale_data=True):
        """
        Give prediction probabilities for the two classes
        """
        if scale_data:
            X = (X - self.x_mean) / self.x_std
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        pred = self.model(X)
        p_y = torch.softmax(pred, dim=1)
        return p_y.cpu().detach().numpy()


    def _acc(self, raw_pred, y_true):
        """
        Used internally to compute accuracy during training without copying out of device
        """
        p_y = torch.softmax(raw_pred, dim=1)
        y_pred = torch.argmax(p_y, dim=1)
        acc = (y_true == y_pred).sum() / len(y_true)
        return acc.item()

    def save(self, dir_path, name):
        mean_std_file_name = f'{dir_path}/{name}_mean_std.npy'
        model_file_name = f'{dir_path}/{name}_model.pt'
        np.save(mean_std_file_name, [self.x_mean, self.x_std])
        torch.save(self.model, model_file_name)

    def load(self, dir_path, name):
        mean_std_file_name = f'{dir_path}/{name}_mean_std.npy'
        model_file_name = f'{dir_path}/{name}_model.pt'
        self.x_mean, self.x_std = np.load(mean_std_file_name)
        self.model = torch.load(model_file_name)
def my_test():
    pass


if __name__ == '__main__':
    my_test()