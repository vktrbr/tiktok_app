import copy
import gc
import os

import dotenv
import ffmpegio
import numpy as np
import torch
from matplotlib import pyplot as plt
from transformers import MobileViTImageProcessor, MobileViTForSemanticSegmentation

dotenv.load_dotenv()


class PreprocessModel(torch.nn.Module):
    device = 'cpu'

    FPS_DIV = 30
    MAX_LENGTH = 90
    BATCH_SIZE = 4
    MAX_TIME = 20.0
    MODEL_PATH = os.getenv('MODEL_PATH')

    def __init__(self):
        super().__init__()
        self.feature_extractor = MobileViTImageProcessor.from_pretrained("apple/deeplabv3-mobilevit-small")
        self.mobile_vit = MobileViTForSemanticSegmentation.from_pretrained("apple/deeplabv3-mobilevit-small")
        self.convs = torch.nn.Sequential(
            torch.nn.MaxPool2d(2, 2)
        )
        self.frames: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mobile_vit(x).logits
        x = self.convs(x)
        return x

    def read_video(self, path: str) -> torch.Tensor:
        self.frames = None
        fps, video = ffmpegio.video.read(path, t=self.MAX_TIME)
        video = video[::int(fps)][:self.MAX_LENGTH]

        out_seg_video = []

        for i in range(0, video.shape[0], self.BATCH_SIZE):
            frames = [video[j] for j in range(i, min(i + self.BATCH_SIZE, video.shape[0]))]
            frames = self.feature_extractor(images=frames, return_tensors='pt')['pixel_values']
            out = self.forward(frames.to(self.device)).detach().to('cpu')
            out_seg_video.append(out)

            if self.frames is None:
                self.frames = frames.detach()
            else:
                self.frames = torch.concat((self.frames, frames.detach()))

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


class TikTokAnalytics(torch.nn.Module):
    """
    A PyTorch module for performing analytics on TikTok videos.
    """

    MAX_FRAMES = 20  # Maximum number of frames to consider in a TikTok video
    NUM_CHANNELS = 21  # Number of channels in the input tensor

    def __init__(self, low_attention: int = -2, high_attention: int = 2, rgw_thr: tuple[float] = (0.4, 0.4, 0.25)):
        """

        :param low_attention:
        :param high_attention:
        :param rgw_thr: red_green_white_thr
        """
        super().__init__()

        self.low_attention = low_attention
        self.high_attention = high_attention
        self.rgw_thr = rgw_thr
        # Initialize preprocessing model and load trained predict model
        self.preprocessing_model = PreprocessModel()
        self.predict_model = torch.load(self.preprocessing_model.MODEL_PATH,
                                        map_location=self.preprocessing_model.device)

        # Initialize upsampling layer and set models to eval mode
        self.upsample = torch.nn.Upsample(scale_factor=512 // 16, mode='bilinear')
        self.preprocessing_model.eval()
        self.predict_model.eval()

        # Initialize frames attribute to None
        self.frames: torch.Tensor | None = None

    def forward(self, path: str) -> torch.Tensor:
        """
        Performs forward pass through the TikTokAnalytics module.

        :param path: Path of the TikTok video to perform analytics on.
        :return: Predicted tensor from the model.
        """

        # Read video using preprocessing model
        tensor = self.preprocessing_model.read_video(path)

        # Make predictions using trained model
        predict = self.predict_model(tensor)

        return predict

    def advanced_forward(self, path: str):
        """
        Performs advanced forward pass through the TikTokAnalytics module.

        :param path: Path of the TikTok video to perform analytics on.
        :return: Output dictionary containing various tensors and figures.
        """
        out_dict = dict()

        # Read video using preprocessing model
        tensor = self.preprocessing_model.read_video(path)
        tensor = tensor.detach()
        upsampler = torch.nn.Upsample(scale_factor=512 // 16, mode='bilinear')

        # Determine number of frames and channels to consider
        num_frames = min(self.MAX_FRAMES, tensor.shape[0])
        num_channels = self.NUM_CHANNELS

        # Make base predictions using trained model and get base like value
        base_predict: torch.Tensor = self.predict_model(tensor)
        base_predict = base_predict.detach()
        base_like_value = base_predict[0, 1].item()

        # Compute channel impact on model's decision
        channel_impact = []
        for i in range(self.NUM_CHANNELS):
            copy_default = copy.deepcopy(tensor)
            copy_default[:, i] = torch.zeros_like(copy_default[:, i])
            channel_impact.append(self.predict_model(copy_default)[0, 1].item())

        raw_channel_impact = channel_impact = torch.tensor(channel_impact)
        ind_max_channel_impact = channel_impact.argsort()[1]

        # Compute frames impact on model's decision
        frames_impact = []
        for j in range(num_frames):
            copy_tensor = torch.concat((tensor[:j], tensor[j + 1:]), dim=0)
            frames_impact.append(self.predict_model(copy_tensor)[0, 1].item())

        raw_frames_impact = frames_impact = torch.tensor(frames_impact)
        ind_max_frame_impact = frames_impact.argmax()

        # Create composite image from all frames and channels
        images = torch.concat(
            [self.preprocessing_model.frames[j] for j in range(num_frames)],
            dim=2
        )
        images = images.repeat(1, num_channels, 1)
        copy_tensor = copy.deepcopy(tensor)
        copy_tensor = upsampler(copy_tensor)

        # Create attention canvas
        canvas = torch.concat(
            [self.minmax(torch.concat([copy_tensor[i, j] for i in range(num_frames)], dim=1), 0.5, 5)
             for j in range(num_channels)],
            dim=0
        )

        # Create masked images based on attention
        masked_images = images.permute(1, 2, 0).flip(2)

        # frames_impact = 2 - frames_impact / base_like_value
        # channel_impact = 2 - channel_impact / base_like_value
        cell_attention_value = frames_impact.reshape(num_frames, 1) * channel_impact.reshape(1, num_channels)

        cell_attention_value = (cell_attention_value * 100 - 100).round()
        cell_attention_value = -(cell_attention_value - torch.median(cell_attention_value.flatten()))
        import streamlit as st
        st.write(self.high_attention)

        for fr_idx in range(num_frames):
            for ch_idx in range(num_channels):
                fi_left, fi_right = fr_idx * 512, (fr_idx + 1) * 512
                ci_left, ci_right = ch_idx * 512, (ch_idx + 1) * 512

                if self.low_attention < cell_attention_value[fr_idx, ch_idx] < self.high_attention:
                    masked_images[fi_left:fi_right, ci_left:ci_right] = (
                            masked_images[fi_left:fi_right, ci_left:ci_right].permute(2, 0, 1) * 1.1
                            * self.whiter_with_threshold(canvas[fi_left:fi_right, ci_left:ci_right], self.rgw_thr[2])
                    ).permute(1, 2, 0)

                elif cell_attention_value[fr_idx, ch_idx] <= self.low_attention:  # red
                    masked_images[fi_left:fi_right, ci_left:ci_right, 0] = (
                            masked_images[fi_left:fi_right, ci_left:ci_right, 0]
                            * self.whiter_with_threshold(canvas[fi_left:fi_right, ci_left:ci_right], self.rgw_thr[0])
                    )

                else:  # green
                    masked_images[fi_left:fi_right, ci_left:ci_right, 1] = (
                            masked_images[fi_left:fi_right, ci_left:ci_right, 1]
                            * self.whiter_with_threshold(canvas[fi_left:fi_right, ci_left:ci_right], self.rgw_thr[1])
                    )

        masked_images = torch.clip(masked_images, 0, 1)

        pad_value = np.ceil(masked_images.shape[1] / 2560)
        pad_value = int(pad_value * 2560 - masked_images.shape[1])
        masked_images = torch.nn.functional.pad(masked_images, (0, 0, 0, pad_value))

        # Create and format pretty filters figure
        pretty_filters = torch.cat([masked_images[
                                    512 * ind_max_channel_impact: 512 * (ind_max_channel_impact + 1),
                                    512 * 5 * i: 512 * 5 * (i + 1)]
                                    for i in range(masked_images.shape[1] // 2560)])

        fig_pretty_filters, ax_pretty_filters = plt.subplots(figsize=(25, 30))
        for ind_frame in range(min(num_frames, 20)):
            (x, y) = (50 + 512 * (ind_frame // 5), 50 + 512 * (ind_frame % 5))
            ax_pretty_filters.text(y, x, f'f:{ind_frame + 1}', color='black',
                                   bbox={'facecolor': 'white', 'edgecolor': 'none',
                                         'boxstyle': 'round'})

        ax_pretty_filters.imshow(pretty_filters)
        ax_pretty_filters.set_axis_off()

        first_20_channels = 20  # This is workaround

        worst_frame = torch.cat([masked_images[
                                 512 * first_20_channels // 5 * i: 512 * first_20_channels // 5 * (i + 1),
                                 512 * ind_max_frame_impact: 512 * (ind_max_frame_impact + 1)]
                                 for i in range(5)
                                 ], dim=1)

        fig_worst_frame, ax_worst_frame = plt.subplots(figsize=(25, 25))
        for ind_frame in range(min(num_frames, 20)):
            (x, y) = (50 + 512 * (ind_frame // 5), 50 + 512 * (ind_frame % 5))
            ax_worst_frame.text(y, x, f'f:{ind_max_frame_impact + 1}', color='black',
                                bbox={'facecolor': 'white', 'edgecolor': 'none', 'boxstyle': 'round'})

        ax_worst_frame.imshow(worst_frame)
        ax_worst_frame.set_axis_off()

        fig_grid, ax_grid = plt.subplots(figsize=(25, 25))
        ax_grid.imshow(masked_images)
        ax_grid.set_axis_off()

        with torch.no_grad():
            # Add output dictionary values
            out_dict['base_predict'] = dict(base=base_predict.tolist(),
                                            sm=(base_predict / base_predict.sum()).flatten().tolist())
            out_dict['channel_impact'] = dict(arr=raw_channel_impact.flatten().tolist(),
                                              ind=int(ind_max_channel_impact))
            out_dict['frames_impact'] = dict(arr=raw_frames_impact.flatten().tolist(),
                                             ind=int(ind_max_frame_impact))
            out_dict['grid_impact'] = cell_attention_value.long().tolist()
            out_dict['pretty_filters'] = dict(fig=fig_pretty_filters, ax=ax_pretty_filters,
                                              tensor=pretty_filters)
            out_dict['grid'] = dict(fig=fig_grid, ax=ax_grid, tensor=masked_images)
            out_dict['worst_frame'] = dict(fig=fig_worst_frame, ax=ax_worst_frame, tensor=worst_frame)

        return out_dict

    @staticmethod
    def sigmoid(t: torch.Tensor):
        """
        Applies sigmoid activation function to a given tensor.

        :param t: Input tensor.
        :return: Tensor with sigmoid activation applied element-wise.
        """
        return (t / (1 + abs(t)) + 1) / 2

    @staticmethod
    def minmax(t: torch.Tensor, loc: float = 0.5, scale: float = 4):
        """
        Performs min-max normalization on a given tensor.

        :param t: Input tensor.
        :param loc: Location parameter for normalization.
        :param scale: Scale parameter for normalization.
        :return: Normalized tensor.
        """
        return ((t - t.min()) / (t.max() - t.min()) - loc) * scale

    @staticmethod
    def whiter_with_threshold(t: torch.Tensor, threshold: float = 0.4):
        """
        Applies a whitening filter to a given tensor, with a threshold applied to retain some structure.

        :param t: Input tensor.
        :param threshold: Threshold value for filter.
        :return: Tensor with filter applied and threshold values clipped.
        """
        t = TikTokAnalytics.sigmoid(t)
        t = torch.clip(t, threshold, 1)
        t[t == threshold] = 0

        return 1 + t / 3


if __name__ == '__main__':
    import os

    os.chdir('..')
    # video_path = '/Users/victorbarbarich/Downloads/video-ds-shock.mp4'
    video_path = '/Users/victorbarbarich/Downloads/IMG_3592.MOV'

    app = TikTokAnalytics()
    out = app.advanced_forward(video_path)
