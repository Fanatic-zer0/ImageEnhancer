"""esrgan_model.py — ESRGAN architecture that supports community checkpoint formats.

Modified from https://github.com/philz1337x/clarity-upscaler
which is a copy of https://github.com/AUTOMATIC1111/stable-diffusion-webui
which is a copy of https://github.com/victorca25/iNNfer
which is a copy of https://github.com/xinntao/ESRGAN

Source: https://huggingface.co/spaces/finegrain/finegrain-image-enhancer (MIT License)
Adapted for use with 4x-UltraSharp and similar community models.
"""

import math
from pathlib import Path
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from PIL import Image


def conv_block(in_nc: int, out_nc: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_nc, out_nc, kernel_size=3, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
    )


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf: int = 64, gc: int = 32) -> None:
        super().__init__()
        self.conv1 = conv_block(nf, gc)
        self.conv2 = conv_block(nf + gc, gc)
        self.conv3 = conv_block(nf + 2 * gc, gc)
        self.conv4 = conv_block(nf + 3 * gc, gc)
        # Wrapped in Sequential because of key in state dict.
        self.conv5 = nn.Sequential(nn.Conv2d(nf + 4 * gc, nf, kernel_size=3, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf: int) -> None:
        super().__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf)
        self.RDB2 = ResidualDenseBlock_5C(nf)
        self.RDB3 = ResidualDenseBlock_5C(nf)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class Upsample2x(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.interpolate(x, scale_factor=2.0)


class ShortcutBlock(nn.Module):
    """Elementwise sum the output of a submodule to its input."""

    def __init__(self, submodule: nn.Module) -> None:
        super().__init__()
        self.sub = submodule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.sub(x)


class RRDBNet(nn.Module):
    def __init__(self, in_nc: int, out_nc: int, nf: int, nb: int) -> None:
        super().__init__()
        assert in_nc % 4 != 0  # in_nc is 3

        self.model = nn.Sequential(
            nn.Conv2d(in_nc, nf, kernel_size=3, padding=1),
            ShortcutBlock(
                nn.Sequential(
                    *(RRDB(nf) for _ in range(nb)),
                    nn.Conv2d(nf, nf, kernel_size=3, padding=1),
                )
            ),
            Upsample2x(),
            nn.Conv2d(nf, nf, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            Upsample2x(),
            nn.Conv2d(nf, nf, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(nf, nf, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(nf, out_nc, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def infer_params(state_dict: dict) -> tuple:
    """Infer RRDBNet architecture params from state dict keys (supports old ESRGAN format)."""
    scale2x = 0
    scalemin = 6
    n_uplayer = 0
    out_nc = 0
    nb = 0

    for block in list(state_dict):
        parts = block.split(".")
        n_parts = len(parts)
        if n_parts == 5 and parts[2] == "sub":
            nb = int(parts[3])
        elif n_parts == 3:
            part_num = int(parts[1])
            if part_num > scalemin and parts[0] == "model" and parts[2] == "weight":
                scale2x += 1
            if part_num > n_uplayer:
                n_uplayer = part_num
                out_nc = state_dict[block].shape[0]
            assert "conv1x1" not in block  # no ESRGANPlus

    nf = state_dict["model.0.weight"].shape[0]
    in_nc = state_dict["model.0.weight"].shape[1]
    scale = 2**scale2x

    assert out_nc > 0
    assert nb > 0

    return in_nc, out_nc, nf, nb, scale  # typically 3, 3, 64, 23, 4


Tile = tuple
Tiles = list


class Grid(NamedTuple):
    tiles: Tiles
    tile_w: int
    tile_h: int
    image_w: int
    image_h: int
    overlap: int


def split_grid(image: Image.Image, tile_w: int = 512, tile_h: int = 512, overlap: int = 64) -> Grid:
    w = image.width
    h = image.height

    non_overlap_width = tile_w - overlap
    non_overlap_height = tile_h - overlap

    cols = max(1, math.ceil((w - overlap) / non_overlap_width))
    rows = max(1, math.ceil((h - overlap) / non_overlap_height))

    dx = (w - tile_w) / (cols - 1) if cols > 1 else 0
    dy = (h - tile_h) / (rows - 1) if rows > 1 else 0

    grid = Grid([], tile_w, tile_h, w, h, overlap)
    for row in range(rows):
        row_images = []
        y1 = max(min(int(row * dy), h - tile_h), 0)
        y2 = min(y1 + tile_h, h)
        for col in range(cols):
            x1 = max(min(int(col * dx), w - tile_w), 0)
            x2 = min(x1 + tile_w, w)
            tile = image.crop((x1, y1, x2, y2))
            row_images.append((x1, tile_w, tile))
        grid.tiles.append((y1, tile_h, row_images))

    return grid


def combine_grid(grid: Grid) -> Image.Image:
    def make_mask_image(r: npt.NDArray[np.float32]) -> Image.Image:
        r = r * 255 / grid.overlap
        return Image.fromarray(r.astype(np.uint8), "L")

    mask_w = make_mask_image(
        np.arange(grid.overlap, dtype=np.float32).reshape((1, grid.overlap)).repeat(grid.tile_h, axis=0)
    )
    mask_h = make_mask_image(
        np.arange(grid.overlap, dtype=np.float32).reshape((grid.overlap, 1)).repeat(grid.image_w, axis=1)
    )

    combined_image = Image.new("RGB", (grid.image_w, grid.image_h))
    for y, h, row in grid.tiles:
        combined_row = Image.new("RGB", (grid.image_w, h))
        for x, w, tile in row:
            if x == 0:
                combined_row.paste(tile, (0, 0))
                continue
            combined_row.paste(tile.crop((0, 0, grid.overlap, h)), (x, 0), mask=mask_w)
            combined_row.paste(tile.crop((grid.overlap, 0, w, h)), (x + grid.overlap, 0))

        if y == 0:
            combined_image.paste(combined_row, (0, 0))
            continue

        combined_image.paste(
            combined_row.crop((0, 0, combined_row.width, grid.overlap)),
            (0, y),
            mask=mask_h,
        )
        combined_image.paste(
            combined_row.crop((0, grid.overlap, combined_row.width, h)),
            (0, y + grid.overlap),
        )

    return combined_image


class UpscalerESRGAN:
    """Loads and runs old-format ESRGAN checkpoints (e.g. 4x-UltraSharp)."""

    def __init__(self, model_path: Path, device: torch.device, dtype: torch.dtype):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.model = self._load_model(model_path)
        self.model.to(device=device, dtype=dtype)

    def to(self, device: torch.device, dtype: torch.dtype) -> None:
        self.device = device
        self.dtype = dtype
        self.model.to(device=device, dtype=dtype)

    def _load_model(self, path: Path) -> RRDBNet:
        state_dict = torch.load(str(path), weights_only=True, map_location="cpu")
        # Some checkpoints wrap weights under a nested key ('params', 'params_ema', etc.)
        for key in ("params_ema", "params", "model", "state_dict"):
            if key in state_dict and isinstance(state_dict[key], dict):
                state_dict = state_dict[key]
                break
        in_nc, out_nc, nf, nb, upscale = infer_params(state_dict)
        assert upscale == 4, f"Only 4x upscaling is supported (got {upscale}x)"
        model = RRDBNet(in_nc=in_nc, out_nc=out_nc, nf=nf, nb=nb)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _upscale_tile(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img.convert("RGB"))
        img_np = img_np[:, :, ::-1]  # RGB → BGR
        img_np = np.ascontiguousarray(np.transpose(img_np, (2, 0, 1))) / 255.0
        img_t = torch.from_numpy(img_np).float().unsqueeze(0).to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            output = self.model(img_t)
        output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = (255.0 * np.moveaxis(output, 0, 2)).astype(np.uint8)
        output = output[:, :, ::-1]  # BGR → RGB
        return Image.fromarray(output, "RGB")

    def upscale_with_tiling(self, img: Image.Image) -> Image.Image:
        img = img.convert("RGB")
        grid = split_grid(img)
        newtiles = []
        scale_factor = 1

        for y, h, row in grid.tiles:
            newrow = []
            for x, w, tile in row:
                output = self._upscale_tile(tile)
                scale_factor = output.width // tile.width
                newrow.append((x * scale_factor, w * scale_factor, output))
            newtiles.append((y * scale_factor, h * scale_factor, newrow))

        newgrid = Grid(
            newtiles,
            grid.tile_w * scale_factor,
            grid.tile_h * scale_factor,
            grid.image_w * scale_factor,
            grid.image_h * scale_factor,
            grid.overlap * scale_factor,
        )
        return combine_grid(newgrid)
