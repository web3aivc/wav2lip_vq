import torch
from torch import nn
from torch.nn import functional as F
from omegaconf import OmegaConf

try:
    from .conv import Conv2d
    from .vqgan import VQModel
except:
    from conv import Conv2d
    from vqgan import VQModel


class SyncNet_color(nn.Module):
    def __init__(self, config_path, ckpt_path=None, syncnet_T=5):
        super(SyncNet_color, self).__init__()
        self.T = syncnet_T

        # (B, 5 x 256, 16, 16) -> (B, 512, 1, 1)
        self.face_encoder = nn.Sequential(
            Conv2d(self.T * 256, 256, kernel_size=3, stride=2, padding=1),  # 16, 16 -> 8, 8
            Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 8, 8 -> 4, 4
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 4, 4 -> 2, 2
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=2, stride=1, padding=0),  # 2, 2 -> 1, 1
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        )

        # (B, 1, 80, 16) -> (B, 512, 1, 1)
        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        )

        config = OmegaConf.load(config_path)
        self.vq_model = VQModel(ckpt_path=ckpt_path, **config.model.params)
        for parameter in self.vq_model.parameters():
            parameter.requires_grad = False
        self.vq_model.eval()

    def forward(self, audio_sequences, face_sequences, vq_encoded=False):
        batch_size = face_sequences.size(0)  # audio_sequences := (B, dim, T)
        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            # (2, 5, 3, 256, 256) -> (2 x 5, 3, 256, 256)
            face_sequences = torch.cat([face_sequences[:, i] for i in range(face_sequences.size(1))], dim=0)
        else:
            batch_size = batch_size // self.T

        # (2 x 5, 3, 256, 256) -> (2 x 5, 256, 16, 16)
        if not vq_encoded:
            face_sequences, _, _ = self.vq_model.encode(face_sequences)

        # (2 x 5, 256, 16, 16) resize (2, 5 x 256, 16, 16)
        # face_sequences = face_sequences.view(batch_size, 5 * 256, 16, 16)
        face_sequences = torch.split(face_sequences, batch_size, dim=0)
        face_sequences = torch.cat(face_sequences, dim=1)

        # (2, 5 x 256, 16, 16) -> (2, 512, 1, 1)
        face_embedding = self.face_encoder(face_sequences)

        # (2, 1, 80, 16) -> (2, 512, 1, 1)
        audio_embedding = self.audio_encoder(audio_sequences)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)  # (2, 512)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)  # (2, 512)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)

        return audio_embedding, face_embedding


if __name__ == '__main__':
    config_path = '../data/vqgan-project.yaml'

    model = SyncNet_color(config_path)
    model.eval()
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    audio_sequences = torch.randn(size=(2, 1, 80, 16))
    face_sequences = torch.randn(size=(2, 5, 3, 256, 256))

    audio_embedding, face_embedding = model(audio_sequences, face_sequences)

    print(audio_embedding.shape)  # (B, 512)
    print(face_embedding.shape)  # (B, 512)
