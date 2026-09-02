from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as error:  # pragma: no cover - exercised only before optional dependency install
    raise ImportError("PyTorch is required for CNN modeling; install torch in the project venv") from error


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = strength
        return input_tensor.view_as(input_tensor)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.strength * gradient, None


class ShapeCNN1D(nn.Module):
    def __init__(
        self,
        input_channels: int,
        class_count: int,
        channels: tuple[int, ...] = (32, 64, 128),
        kernel_sizes: tuple[int, ...] = (15, 9, 5),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if len(channels) != len(kernel_sizes) or not channels:
            raise ValueError("channels and kernel_sizes must be non-empty and have equal length")
        blocks: list[nn.Module] = []
        previous = input_channels
        for width, kernel in zip(channels, kernel_sizes, strict=True):
            blocks.extend(
                [
                    nn.Conv1d(previous, width, kernel_size=kernel, padding=kernel // 2, bias=False),
                    nn.BatchNorm1d(width),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(kernel_size=2),
                ]
            )
            previous = width
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(previous, class_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class ResidualBlock1D(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, dropout: float) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, 7, padding=3, bias=False),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(output_channels, output_channels, 5, padding=2, bias=False),
            nn.BatchNorm1d(output_channels),
        )
        self.skip = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv1d(input_channels, output_channels, 1, bias=False)
        )
        self.activation = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.activation(self.body(x) + self.skip(x)))


class MultiScaleResidualShapeCNN(nn.Module):
    def __init__(
        self,
        input_channels: int,
        class_count: int,
        stem_channels: int = 32,
        branch_channels: int = 24,
        residual_channels: tuple[int, ...] = (72, 96),
        kernel_sizes: tuple[int, ...] = (5, 15, 31),
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, stem_channels, 7, padding=3, bias=False),
            nn.BatchNorm1d(stem_channels),
            nn.ReLU(inplace=True),
        )
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(stem_channels, branch_channels, kernel, padding=kernel // 2, bias=False),
                    nn.BatchNorm1d(branch_channels),
                    nn.ReLU(inplace=True),
                )
                for kernel in kernel_sizes
            ]
        )
        width = branch_channels * len(kernel_sizes)
        blocks: list[nn.Module] = []
        for output_channels in residual_channels:
            blocks.append(ResidualBlock1D(width, output_channels, dropout / 2.0))
            width = output_channels
        self.residual = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width, class_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stem = self.stem(x)
        combined = torch.cat([branch(stem) for branch in self.branches], dim=1)
        return self.head(self.residual(combined))


class SmallCNNEncoder(nn.Module):
    def __init__(self, input_channels: int = 4, widths: tuple[int, ...] = (32, 64, 128), normalization: str = "batch") -> None:
        super().__init__()
        kernels = (15, 9, 5)
        layers: list[nn.Module] = []
        previous = input_channels
        for width, kernel in zip(widths, kernels, strict=True):
            norm = nn.BatchNorm1d(width) if normalization == "batch" else nn.GroupNorm(min(8,width),width)
            layers.extend([
                nn.Conv1d(previous, width, kernel, padding=kernel // 2, bias=False),
                norm, nn.ReLU(inplace=True), nn.MaxPool1d(2),
            ])
            previous = width
        self.network = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.output_dim = widths[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class _TCNResidualBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float = 0.1) -> None:
        super().__init__()
        padding = dilation * 3
        self.body = nn.Sequential(
            nn.Conv1d(width, width, 7, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(width, width, 7, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(width),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.body(x))


class TCNEncoder(nn.Module):
    def __init__(self, input_channels: int = 4, width: int = 48) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_channels, width, 7, padding=3, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            _TCNResidualBlock(width, 1),
            _TCNResidualBlock(width, 2),
            _TCNResidualBlock(width, 4),
            _TCNResidualBlock(width, 8),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.output_dim = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class EEGNetEncoder(nn.Module):
    """Compact temporal-then-channel encoder adapted to four sEMG channels."""

    def __init__(self, input_channels: int = 4) -> None:
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, 8, (1, 63), padding=(0, 31), bias=False),
            nn.BatchNorm2d(8),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(8, 16, (input_channels, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(inplace=True),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.25),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(16, 16, (1, 15), padding=(0, 7), groups=16, bias=False),
            nn.Conv2d(16, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.25),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.output_dim = 32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.separable(self.spatial(self.temporal(x.unsqueeze(1))))


class _InceptionModule1D(nn.Module):
    def __init__(self, input_channels: int, bottleneck_channels: int = 32, branch_channels: int = 16) -> None:
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv1d(input_channels, bottleneck_channels, 1, bias=False),
            nn.BatchNorm1d(bottleneck_channels),
            nn.ReLU(inplace=True),
        )
        self.branches = nn.ModuleList([
            nn.Conv1d(bottleneck_channels, branch_channels, kernel, padding=kernel // 2, bias=False)
            for kernel in (15, 31, 63, 127)
        ])
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(input_channels, branch_channels, 1, bias=False),
        )
        self.normalization = nn.BatchNorm1d(branch_channels * 5)
        self.activation = nn.ReLU(inplace=True)
        self.output_dim = branch_channels * 5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reduced = self.bottleneck(x)
        combined = torch.cat([branch(reduced) for branch in self.branches] + [self.pool_branch(x)], dim=1)
        return self.activation(self.normalization(combined))


class _ChannelAttention1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.network = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.network(x)


class InceptionTimeEncoder(nn.Module):
    """Parallel multi-scale temporal convolutions with residual and channel attention."""

    def __init__(self, input_channels: int = 4) -> None:
        super().__init__()
        first = _InceptionModule1D(input_channels)
        second = _InceptionModule1D(first.output_dim)
        third = _InceptionModule1D(second.output_dim)
        fourth = _InceptionModule1D(third.output_dim)
        width = fourth.output_dim
        self.first_group = nn.Sequential(first, second)
        self.first_skip = nn.Sequential(nn.Conv1d(input_channels, width, 1, bias=False), nn.BatchNorm1d(width))
        self.second_group = nn.Sequential(third, fourth)
        self.attention = _ChannelAttention1D(width)
        self.activation = nn.ReLU(inplace=True)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.output_dim = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.activation(self.first_group(x) + self.first_skip(x))
        encoded = self.activation(self.second_group(encoded) + encoded)
        return self.pool(self.attention(encoded))


class STFTEncoder(nn.Module):
    def __init__(self, input_channels: int = 4, n_fft: int = 128, hop_length: int = 32) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.output_dim = 32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, length = x.shape
        with torch.autocast(device_type=x.device.type, enabled=False):
            spectrum = torch.stft(
                x.float().reshape(batch * channels, length),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=self.window,
                center=False,
                return_complex=True,
            )
            spectrum = torch.log1p(spectrum.abs()).reshape(batch, channels, spectrum.shape[-2], spectrum.shape[-1])
        return self.network(spectrum)


class ShapeVariantModel(nn.Module):
    """Five controlled shape-model variants sharing the same small CNN backbone."""

    def __init__(
        self,
        variant: str,
        feature_count: int = 80,
        dropout: float = 0.3,
        input_channels: int = 4,
    ) -> None:
        super().__init__()
        self.variant = variant
        if variant == "tcn_mixup":
            self.raw_encoder = TCNEncoder(input_channels=input_channels)
        elif variant == "eegnet_mixup":
            self.raw_encoder = EEGNetEncoder(input_channels=input_channels)
        elif variant == "inceptiontime_mixup":
            self.raw_encoder = InceptionTimeEncoder(input_channels=input_channels)
        else:
            self.raw_encoder = SmallCNNEncoder(
                input_channels=input_channels,
                normalization="group" if variant == "size_specific_groupnorm" else "batch",
            )
        width = self.raw_encoder.output_dim
        if variant == "ordinary":
            self.shape_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, 3))
        elif variant == "size_embedding":
            self.size_embedding = nn.Embedding(6, 8)
            self.shape_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(width + 8, 3))
        elif variant in {"size_film", "size_hand_film"}:
            condition_count = 6 if variant == "size_film" else 12
            self.condition_embedding = nn.Embedding(condition_count, 16)
            self.film = nn.Linear(16, width * 2)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
            self.shape_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, 3))
        elif variant == "size_hand_embedding":
            self.hand_embedding = nn.Embedding(2, 4)
            self.size_heads = nn.ModuleList(
                [nn.Sequential(nn.Dropout(dropout), nn.Linear(width + 4, 3)) for _ in range(6)]
            )
        elif variant == "multitask":
            self.shape_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, 3))
            self.size_ordinal_head = nn.Linear(width, 5)
        elif variant in {"size_specific_heads", "size_specific_supcon", "size_specific_groupnorm", "size_specific_coral", "size_specific_adversarial", "size_specific_label_smoothing", "size_specific_mixup", "size_specific_time_shift", "size_specific_adabn", "file_ssl_pretrain", "size_specific_mixstyle", "tcn_mixup", "eegnet_mixup", "inceptiontime_mixup"}:
            self.size_heads = nn.ModuleList([nn.Sequential(nn.Dropout(dropout), nn.Linear(width, 3)) for _ in range(6)])
            if variant == "size_specific_adversarial":
                self.participant_head = nn.Sequential(
                    nn.Linear(width, 64), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(64, 15)
                )
        elif variant == "shared_size_residual_mixup":
            self.shared_shape_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, 3))
            self.size_residual_down = nn.ModuleList([nn.Linear(width, 8, bias=False) for _ in range(6)])
            self.size_residual_up = nn.ModuleList([nn.Linear(8, 3, bias=False) for _ in range(6)])
            for layer in self.size_residual_up:
                nn.init.zeros_(layer.weight)
        elif variant == "size_group_experts":
            self.group_experts=nn.ModuleList([nn.Sequential(nn.Linear(width,width),nn.ReLU(inplace=True),nn.Dropout(dropout)) for _ in range(3)])
            self.size_heads=nn.ModuleList([nn.Linear(width,3) for _ in range(6)])
        elif variant == "stft_dual":
            self.stft_encoder = STFTEncoder(input_channels=input_channels)
            self.size_heads = nn.ModuleList(
                [nn.Sequential(nn.Dropout(dropout), nn.Linear(width + self.stft_encoder.output_dim, 3)) for _ in range(6)]
            )
        elif variant == "size_hand_specific_heads":
            self.size_hand_heads = nn.ModuleList(
                [nn.Sequential(nn.Dropout(dropout), nn.Linear(width, 3)) for _ in range(12)]
            )
        elif variant in {"dual_representation", "phase_dual", "envelope_dual"}:
            self.normalized_encoder = SmallCNNEncoder(input_channels=input_channels)
            self.shape_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(width * 2, 3))
        elif variant == "feature_fusion":
            self.feature_mlp = nn.Sequential(nn.Linear(feature_count, 64), nn.ReLU(inplace=True), nn.Dropout(dropout))
            self.shape_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(width + 64, 3))
        else:
            raise ValueError(f"Unknown shape variant {variant!r}")

    def forward(
        self,
        raw: torch.Tensor,
        size_index: torch.Tensor,
        normalized: torch.Tensor | None = None,
        handcrafted: torch.Tensor | None = None,
        hand_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        embedding = self.raw_encoder(raw)
        auxiliary = None
        if self.variant == "ordinary":
            logits = self.shape_head(embedding)
        elif self.variant == "size_embedding":
            logits = self.shape_head(torch.cat([embedding, self.size_embedding(size_index)], dim=1))
        elif self.variant in {"size_film", "size_hand_film"}:
            if self.variant == "size_hand_film":
                if hand_index is None:
                    raise ValueError("size_hand_film requires hand_index")
                condition_index = size_index * 2 + hand_index
            else:
                condition_index = size_index
            scale, shift = self.film(self.condition_embedding(condition_index)).chunk(2, dim=1)
            conditioned = embedding * (1.0 + scale) + shift
            logits = self.shape_head(conditioned)
        elif self.variant == "size_hand_embedding":
            if hand_index is None:
                raise ValueError("size_hand_embedding requires hand_index")
            combined = torch.cat([embedding, self.hand_embedding(hand_index)], dim=1)
            all_logits = torch.stack([head(combined) for head in self.size_heads], dim=1)
            logits = all_logits[torch.arange(len(size_index), device=size_index.device), size_index]
        elif self.variant == "multitask":
            logits = self.shape_head(embedding)
            auxiliary = self.size_ordinal_head(embedding)
        elif self.variant in {"size_specific_heads", "size_specific_supcon", "size_specific_groupnorm", "size_specific_coral", "size_specific_adversarial", "size_specific_label_smoothing", "size_specific_mixup", "size_specific_time_shift", "size_specific_adabn", "file_ssl_pretrain", "size_specific_mixstyle", "tcn_mixup", "eegnet_mixup", "inceptiontime_mixup"}:
            if self.variant == "size_specific_mixstyle" and self.training and bool(torch.rand((), device=embedding.device) < 0.5):
                mean = embedding.mean(dim=1, keepdim=True)
                std = embedding.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
                normalized_embedding = (embedding - mean) / std
                permutation = torch.randperm(len(embedding), device=embedding.device)
                mixing = torch.distributions.Beta(0.1, 0.1).sample((len(embedding), 1)).to(embedding.device)
                embedding = normalized_embedding * (mixing * std + (1.0 - mixing) * std[permutation])
                embedding = embedding + mixing * mean + (1.0 - mixing) * mean[permutation]
            all_logits = torch.stack([head(embedding) for head in self.size_heads], dim=1)
            logits = all_logits[torch.arange(len(size_index), device=size_index.device), size_index]
            if self.variant in {"size_specific_supcon", "size_specific_coral", "file_ssl_pretrain"}:
                auxiliary = embedding
            elif self.variant == "size_specific_adversarial":
                auxiliary = self.participant_head(_GradientReversal.apply(embedding, 0.1))
        elif self.variant == "shared_size_residual_mixup":
            shared_logits = self.shared_shape_head(embedding)
            all_residuals = torch.stack(
                [
                    up(torch.relu(down(embedding)))
                    for down, up in zip(self.size_residual_down, self.size_residual_up, strict=True)
                ],
                dim=1,
            )
            logits = shared_logits + all_residuals[
                torch.arange(len(size_index), device=size_index.device), size_index
            ]
        elif self.variant == "size_group_experts":
            group=torch.where(size_index<=1,0,torch.where(size_index<=4,1,2))
            expert_all=torch.stack([expert(embedding) for expert in self.group_experts],1)
            adapted=embedding+expert_all[torch.arange(len(group),device=group.device),group]
            all_logits=torch.stack([head(adapted) for head in self.size_heads],1)
            logits=all_logits[torch.arange(len(size_index),device=size_index.device),size_index]
        elif self.variant == "stft_dual":
            combined = torch.cat([embedding, self.stft_encoder(raw)], dim=1)
            all_logits = torch.stack([head(combined) for head in self.size_heads], dim=1)
            logits = all_logits[torch.arange(len(size_index), device=size_index.device), size_index]
        elif self.variant == "size_hand_specific_heads":
            if hand_index is None:
                raise ValueError("size_hand_specific_heads requires hand_index")
            head_index = size_index * 2 + hand_index
            all_logits = torch.stack([head(embedding) for head in self.size_hand_heads], dim=1)
            logits = all_logits[torch.arange(len(head_index), device=head_index.device), head_index]
        elif self.variant in {"dual_representation", "phase_dual", "envelope_dual"}:
            if normalized is None:
                raise ValueError("dual_representation requires normalized input")
            logits = self.shape_head(torch.cat([embedding, self.normalized_encoder(normalized)], dim=1))
        else:
            if handcrafted is None:
                raise ValueError("feature_fusion requires handcrafted input")
            logits = self.shape_head(torch.cat([embedding, self.feature_mlp(handcrafted)], dim=1))
        return logits, auxiliary
