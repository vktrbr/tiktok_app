import gc
import os

import dotenv
import ffmpegio
import streamlit as st
import torch
from transformers import MobileViTImageProcessor, MobileViTForSemanticSegmentation

dotenv.load_dotenv()


class PreprocessModel(torch.nn.Module):
    device = 'cpu'

    FPS_DIV = 3
    MAX_LENGTH = 90
    BATCH_SIZE = 4
    MODEL_PATH = os.getenv('MODEL_PATH')

    def __init__(self):
        super().__init__()
        self.feature_extractor = MobileViTImageProcessor.from_pretrained("apple/deeplabv3-mobilevit-small")
        self.mobile_vit = MobileViTForSemanticSegmentation.from_pretrained("apple/deeplabv3-mobilevit-small")
        self.convs = torch.nn.Sequential(
            torch.nn.MaxPool2d(2, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mobile_vit(x).logits
        x = self.convs(x)
        return x

    def read_video(self, path: str) -> torch.Tensor:

        _, video = ffmpegio.video.read(path, t=1.0)
        video = video[::self.FPS_DIV][:self.MAX_LENGTH]

        out_seg_video = []

        for i in range(0, video.shape[0], self.BATCH_SIZE):
            frames = [video[j] for j in range(i, min(i + self.BATCH_SIZE, video.shape[0]))]
            frames = self.feature_extractor(images=frames, return_tensors='pt')['pixel_values']

            out = self.forward(frames.to(self.device)).detach().to('cpu')
            out_seg_video.append(out)

            del frames, out
            gc.collect()
            if self.device == 'cuda':
                torch.cuda.empty_cache()

        return torch.cat(out_seg_video)


class VideoModel(torch.nn.Module):
    MAX_LENGTH = 90

    def __init__(self):
        super().__init__()
        p = 0.5
        self.pic_cnn = torch.nn.Sequential(
            torch.nn.Conv2d(21, 128, (2, 2), stride=2),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(128, 256, (2, 2), stride=2),
            torch.nn.BatchNorm2d(256),
            torch.nn.Dropout2d(p),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(256, 256, (4, 4), stride=2),
            torch.nn.BatchNorm2d(256),
            torch.nn.Dropout2d(p),
            torch.nn.Flatten()
        )

        self.vid_cnn = torch.nn.Sequential(
            torch.nn.Conv2d(21, 128, (2, 2), stride=2),
            torch.nn.BatchNorm2d(128),
            torch.nn.Tanh(),
            torch.nn.Conv2d(128, 256, (2, 2), stride=2),
            torch.nn.BatchNorm2d(256),
            torch.nn.Dropout2d(p),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(256, 512, (2, 2), stride=2),
            torch.nn.BatchNorm2d(512),
            torch.nn.Dropout2d(p),
            torch.nn.Flatten()
        )

        self.lstm = torch.nn.LSTM(2048, 256, 1, batch_first=True, bidirectional=True)
        self.fc1 = torch.nn.Linear(256 * 2, 1024)
        self.fc_norm = torch.nn.BatchNorm1d(256 * 2)
        self.tanh = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(1024, 2)
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(p)

        # xaiver init
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv3d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        :param video: torch.Tensor, shape = (batch_size, frames + 1, 1344)
        """
        frames = video.shape[0]
        video = torch.nn.functional.pad(video, (0, 0, 0, 0, 0, 0, self.MAX_LENGTH + 1 - frames, 0))
        video = video.unsqueeze(0)
        _batch_size = video.shape[0]

        _preview = video[:, 0, :, :]
        _video = video[:, 1:, :, :]

        h0 = self.pic_cnn(_preview).unsqueeze(0)
        h0 = torch.nn.functional.pad(h0, (0, 0, 0, 0, 0, 1))
        c0 = torch.zeros_like(h0)

        _video = self.vid_cnn(_video.reshape(-1, 21, 16, 16))
        _video = _video.reshape(_batch_size, 90, -1)

        context, _ = self.lstm(_video, (h0, c0))
        out = self.fc_norm(context[:, -1])
        out = self.tanh(self.fc1(out))
        out = self.dropout(out)
        out = self.sigmoid(self.fc2(out))
        return out


@st.cache_resource
class TikTokAnalytics(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.preprocessing_model = PreprocessModel()
        self.predict_model = torch.load(self.preprocessing_model.MODEL_PATH,
                                        map_location=self.preprocessing_model.device)

        self.preprocessing_model.eval()
        self.predict_model.eval()

    def forward(self, path: str) -> torch.Tensor:
        """
        :param path:
        :return:
        """
        tensor = self.preprocessing_model.read_video(path)
        predict = self.predict_model(tensor)

        return predict
