import torch
import torch.nn as nn
import torch.nn.functional as F


def swish(x, beta=1):
    return x * torch.nn.Sigmoid()(x * beta)


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return F.tanh(grad_output)


class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x


class WaveAct(nn.Module):
    def __init__(self):
        super(WaveAct, self).__init__()
        self.w1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        return self.w1 * torch.sin(x) + self.w2 * torch.cos(x)


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_output = self.decoder(z)
        return x_output, z


class PINNs_CONV(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, failure_matrix, prob_failure, type):
        super(PINNs_CONV, self).__init__()

        self.type = type

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        failure_matrix = torch.FloatTensor(failure_matrix).to(device)
        prob_failure = torch.FloatTensor(prob_failure).to(device)

        self.N_failure = failure_matrix.shape[0]
        self.Ng = failure_matrix.shape[1]
        self.failure_matrix = failure_matrix
        self.prob_failure = prob_failure.squeeze()

        self.STE = StraightThroughEstimator()

        self.ln = nn.LayerNorm(self.N_failure)

        latent_dim = hidden_dim
        self.Autoencoder = Autoencoder(self.N_failure, 2 * hidden_dim, latent_dim)

        kernel_size1 = 3
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size1, stride=1,
                               padding=(kernel_size1 - 1) // 2)

        kernel_size2 = 3
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size2, stride=1,
                               padding=(kernel_size2 - 1) // 2)

        kernel_size3 = 3
        self.conv3 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size3, stride=1,
                               padding=(kernel_size3 - 1) // 2)

        kernel_size4 = 3
        self.conv4 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size4, stride=1,
                               padding=(kernel_size4 - 1) // 2)

        kernel_size5 = 3
        self.conv5 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size5, stride=1,
                               padding=(kernel_size5 - 1) // 2)

        kernel_size6 = 3
        self.conv6 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size6, stride=1,
                               padding=(kernel_size6 - 1) // 2)

        kernel_size7 = 3
        self.conv7 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size7, stride=1,
                               padding=(kernel_size7 - 1) // 2)

        kernel_size8 = 3
        self.conv8 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size8, stride=1,
                               padding=(kernel_size8 - 1) // 2)

        kernel_size9 = 3
        self.conv9 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size9, stride=1,
                               padding=(kernel_size9 - 1) // 2)

        self.flat = nn.Flatten(start_dim=1)

        self.fc1 = nn.Linear(latent_dim + in_dim, 4 * hidden_dim)
        self.fc2 = nn.Linear(4 * hidden_dim, 2 * hidden_dim)
        self.fc3 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, out_dim)

        self.act1 = WaveAct()
        self.act2 = WaveAct()
        self.act3 = WaveAct()

    def forward(self, x):
        x1 = torch.sum(x[:, self.Ng:], dim=1).unsqueeze(1)
        x2 = torch.matmul(x[:, 0:self.Ng], self.failure_matrix.T)

        x0 = x1 - x2

        if self.type == 1:
            x_original = self.ln(F.leaky_relu(x0, negative_slope=0.01)  * self.prob_failure)
            # x_original = self.ln(swish(x0) * self.prob_failure)
        else:
            x_original = self.ln(self.STE(x0) * self.prob_failure)

        x_decoded, x_encoded = self.Autoencoder(x_original)

        x_failure = x_encoded

        x = torch.cat([x, x_failure], 1)

        x = x.unsqueeze(1)

        res = x
        x = swish(self.conv1(x))
        x = swish(self.conv2(x))
        x = swish(self.conv3(x))
        x += res

        res = x
        x = swish(self.conv4(x))
        x = swish(self.conv5(x))
        x = swish(self.conv6(x))
        x += res

        res = x
        x = swish(self.conv7(x))
        x = swish(self.conv8(x))
        x = swish(self.conv9(x))
        x += res

        x = self.flat(x)

        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        x = self.fc4(x)

        return x[:, 0], x_decoded, x_original
