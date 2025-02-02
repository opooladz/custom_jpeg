import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from PIL import Image
import io
import numpy as np
import random
import os
from scipy.fftpack import dct, idct  # Import DCT and IDCT functions
from scipy.ndimage import zoom
import cv2  # Ensure OpenCV is installed (pip install opencv-python)

# Standard JPEG Compression Function using PIL
def jpeg_compress(img, quality=100):
    buffer = io.BytesIO()
    img_pil = T.ToPILImage()(img.cpu())
    img_pil.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    img_compressed = Image.open(buffer)
    img_compressed = img_compressed.resize(img_pil.size)  # Ensure same size
    return T.ToTensor()(img_compressed).to(img.device)

# Custom JPEG Compression using block-wise DCT and quantization (improved version)
def jpeg_compress_custom(img, quality=95):
    """
    Improved custom JPEG compression that:
      1. Converts RGB to YCbCr.
      2. Uses quality-dependent scaling of separate quantization matrices for Y (luminance)
         and Cb/Cr (chrominance) channels.
      3. Applies blockwise DCT, quantization, dequantization, and inverse DCT.
      4. Converts back to RGB.
      
    Parameters:
      img: Torch tensor of shape [C, H, W] with pixel values in [0, 1].
      quality: Integer quality factor (typically between 1 and 100).
      
    Returns:
      A torch tensor of shape [C, H, W] representing the compressed image.
    """
    QY = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])
    QC = np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99]
    ])
    
    quality = max(1, min(quality, 100))
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - quality * 2

    QY_scaled = np.clip(np.floor((QY * scale + 50) / 100), 1, 255)
    QC_scaled = np.clip(np.floor((QC * scale + 50) / 100), 1, 255)
    
    def blockwise_dct(block):
        return dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
    def blockwise_idct(block):
        return idct(idct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
    
    img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.float32)
    h, w, _ = img_np.shape

    R = img_np[..., 0]
    G = img_np[..., 1]
    B = img_np[..., 2]
    Y_val  =  0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128
    Cr =  0.5 * R - 0.418688 * G - 0.081312 * B + 128

    img_ycbcr = np.stack([Y_val, Cb, Cr], axis=-1)
    img_dct = np.zeros_like(img_ycbcr)
    
    for channel in range(3):
        Q_current = QY_scaled if channel == 0 else QC_scaled
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                block = img_ycbcr[i:i+8, j:j+8, channel]
                if block.shape[0] < 8 or block.shape[1] < 8:
                    pad_h = 8 - block.shape[0]
                    pad_w = 8 - block.shape[1]
                    block = np.pad(block, ((0, pad_h), (0, pad_w)), mode='edge')
                dct_block = blockwise_dct(block)
                quant_block = np.round(dct_block / Q_current)
                idct_block = blockwise_idct(quant_block * Q_current)
                idct_block = idct_block[:block.shape[0], :block.shape[1]]
                img_dct[i:i+8, j:j+8, channel] = idct_block

    Y_recon  = img_dct[..., 0]
    Cb_recon = img_dct[..., 1] - 128
    Cr_recon = img_dct[..., 2] - 128
    R_rec = Y_recon + 1.402 * Cr_recon
    G_rec = Y_recon - 0.344136 * Cb_recon - 0.714136 * Cr_recon
    B_rec = Y_recon + 1.772 * Cb_recon
    img_rgb = np.stack([R_rec, G_rec, B_rec], axis=-1)
    img_rgb = np.clip(img_rgb, 0, 255)
    
    img_recon = torch.tensor(img_rgb / 255.0).permute(2, 0, 1).clamp(0, 1)
    return img_recon

# Variant with 4:2:0 chroma subsampling (for our visualization)
def jpeg_compress_custom_with_subsampling(img, quality=95):
    img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.float32)
    H, W, _ = img_np.shape

    R = img_np[:, :, 0]
    G = img_np[:, :, 1]
    B = img_np[:, :, 2]
    Y_val = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128
    Cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 128

    quality = max(1, min(quality, 100))
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - quality * 2

    QY = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])
    QC = np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99]
    ])
    
    QY_scaled = np.clip(np.floor((QY * scale + 50) / 100), 1, 255)
    QC_scaled = np.clip(np.floor((QC * scale + 50) / 100), 1, 255)
    
    def process_channel(channel, Q):
        h, w = channel.shape
        out = np.zeros_like(channel)
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                block = channel[i:i+8, j:j+8]
                if block.shape[0] < 8 or block.shape[1] < 8:
                    pad_h = 8 - block.shape[0]
                    pad_w = 8 - block.shape[1]
                    block = np.pad(block, ((0, pad_h), (0, pad_w)), mode='edge')
                dct_block = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
                quant_block = np.round(dct_block / Q)
                idct_block = idct(idct(quant_block * Q, axis=0, norm='ortho'), axis=1, norm='ortho')
                out[i:i+8, j:j+8] = idct_block[:block.shape[0], :block.shape[1]]
        return out

    # Process Y channel (full resolution)
    Y_centered = Y_val - 128
    Y_processed = process_channel(Y_centered, QY_scaled) + 128

    # Process chrominance channels with 4:2:0 subsampling
    Cb_ds = Cb[::2, ::2]
    Cr_ds = Cr[::2, ::2]
    Cb_ds_centered = Cb_ds - 128
    Cr_ds_centered = Cr_ds - 128
    Cb_processed_ds = process_channel(Cb_ds_centered, QC_scaled) + 128
    Cr_processed_ds = process_channel(Cr_ds_centered, QC_scaled) + 128
    Cb_processed = zoom(Cb_processed_ds, zoom=2, order=1)[:H, :W]
    Cr_processed = zoom(Cr_processed_ds, zoom=2, order=1)[:H, :W]

    img_ycbcr = np.stack([Y_processed, Cb_processed, Cr_processed], axis=-1)
    Y_rec = img_ycbcr[:, :, 0]
    Cb_rec = img_ycbcr[:, :, 1] - 128
    Cr_rec = img_ycbcr[:, :, 2] - 128
    R_rec = Y_rec + 1.402 * Cr_rec
    G_rec = Y_rec - 0.344136 * Cb_rec - 0.714136 * Cr_rec
    B_rec = Y_rec + 1.772 * Cb_rec
    img_rgb = np.stack([R_rec, G_rec, B_rec], axis=-1)
    img_rgb = np.clip(img_rgb, 0, 255)
    
    return torch.tensor(img_rgb / 255.0).permute(2, 0, 1).clamp(0, 1)

# Deblocking function using bilateral filter from OpenCV
def deblock_image(img_tensor, diameter=2, sigma_color=25, sigma_space=75):
    img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    deblocked_np = cv2.bilateralFilter(img_np, d=diameter, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    return torch.tensor(deblocked_np.astype(np.float32) / 255.0).permute(2, 0, 1)

# Dithering function using an ordered dithering approach with a Bayer matrix.
def apply_dither(img, bit_depth=8):
    levels = 2 ** bit_depth
    bayer = np.array([[0,  8,  2, 10],
                      [12, 4, 14, 6],
                      [3, 11, 1, 9],
                      [15, 7, 13, 5]], dtype=np.float32)
    bayer = (bayer + 0.5) / 16.0
    C, H, W = img.shape
    bayer_tile = np.tile(bayer, (int(np.ceil(H / 4)), int(np.ceil(W / 4))))[:H, :W]
    img_np = img.cpu().numpy()
    dithered = np.zeros_like(img_np)
    for c in range(C):
        channel = img_np[c]
        channel_scaled = channel * (levels - 1)
        dithered_channel = np.floor(channel_scaled + bayer_tile)
        dithered_channel = np.clip(dithered_channel, 0, levels - 1)
        dithered[c] = dithered_channel / (levels - 1)
    return torch.tensor(dithered, dtype=img.dtype, device=img.device)

# Visualization function that computes and displays the different JPEG variants.
def visualize(original_img, perturbed_img):
    # Convert from the normalized [-1, 1] range to [0, 1] for display.
    original_disp = (original_img.permute(1, 2, 0).cpu() + 1) / 2
    perturbed_disp = (perturbed_img.permute(1, 2, 0).cpu() + 1) / 2

    # Compute variants using the perturbed image.
    standard_jpeg = jpeg_compress(perturbed_disp.permute(2, 0, 1))
    custom_jpeg = jpeg_compress_custom_with_subsampling(perturbed_disp.permute(2, 0, 1))
    deblocked_custom = deblock_image(custom_jpeg)
    custom_deblock_dither = apply_dither(deblocked_custom)

    mse_loss = nn.MSELoss()
    mse_perturbed = mse_loss(perturbed_disp, original_disp).item()
    mse_standard = mse_loss(standard_jpeg.permute(1,2,0), original_disp).item()
    mse_custom = mse_loss(custom_jpeg.permute(1,2,0), original_disp).item()
    mse_deblocked = mse_loss(deblocked_custom.permute(1,2,0), original_disp).item()
    mse_dithered = mse_loss(custom_deblock_dither.permute(1,2,0), original_disp).item()

    # Display six columns:
    # 1. Original, 2. Perturbed, 3. Standard JPEG, 4. Custom JPEG,
    # 5. Deblocked Custom JPEG, 6. Deblocked + Dithered Custom JPEG.
    plt.figure(figsize=(70, 5))
    plt.subplot(1, 6, 1)
    plt.imshow(original_disp.numpy())
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 6, 2)
    plt.imshow(perturbed_disp.numpy())
    plt.title(f'Perturbed\nMSE: {mse_perturbed:.4f}')
    plt.axis('off')
    
    plt.subplot(1, 6, 3)
    plt.imshow(standard_jpeg.permute(1,2,0).cpu().numpy())
    plt.title(f'Standard JPEG\nMSE: {mse_standard:.4f}')
    plt.axis('off')
    
    plt.subplot(1, 6, 4)
    plt.imshow(custom_jpeg.permute(1,2,0).cpu().numpy())
    plt.title(f'Custom JPEG\nMSE: {mse_custom:.4f}')
    plt.axis('off')
    
    plt.subplot(1, 6, 5)
    plt.imshow(deblocked_custom.permute(1,2,0).cpu().numpy())
    plt.title(f'Deblocked Custom\nMSE: {mse_deblocked:.4f}')
    plt.axis('off')
    
    plt.subplot(1, 6, 6)
    plt.imshow(custom_deblock_dither.permute(1,2,0).cpu().numpy())
    plt.title(f'Deblocked + Dithered\nMSE: {mse_dithered:.4f}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

    analyze_frequency_spectrum(original_disp.permute(2, 0, 1), perturbed_disp.permute(2, 0, 1))

# Frequency Spectrum Analysis
def analyze_frequency_spectrum(original_img, perturbed_img):
    difference = perturbed_img - original_img
    # Compute the mean across color channels to obtain a grayscale difference image.
    diff_gray = difference.mean(dim=0).cpu().numpy()
    fft_result = np.fft.fftshift(np.fft.fft2(diff_gray))
    magnitude_spectrum = np.log(np.abs(fft_result) + 1e-8)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(diff_gray, cmap='gray')
    plt.title('Perturbation (Difference Image)')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(magnitude_spectrum, cmap='hot')
    plt.title('Frequency Spectrum')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Main Processing Loop
def process_images():
    # Set seeds for reproducibility.
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load an additional tensor from a file (assuming this tensor provides the perturbation)
    data = torch.load('/content/0.pth', map_location='cpu')
    _, _, image_tensor = data
    additional_tensor = torch.tensor(image_tensor[0]).to(device)

    # Define a transform to convert CIFAR10 images to the required format: 32x32 and pixel range [-1, 1].
    transform = T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
        T.Lambda(lambda x: (x * 2) - 1)
    ])
    cifar10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(cifar10, batch_size=1, shuffle=True)

    # Process a few images.
    for i, (img, _) in enumerate(dataloader):
        img = img.squeeze(0).to(device)
        # Create a perturbed image by adding the additional tensor (and clipping to [-1, 1])
        perturbed_img = torch.clamp(img + additional_tensor, -1, 1)
        visualize(img, perturbed_img)
        if i == 4:
            break

if __name__ == "__main__":
    process_images()
