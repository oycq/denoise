# ==================== 1. DPI 缩放 + 环境设置 ====================
import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)  # 1 = System DPI Aware
except:
    pass

import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import wandb

# ==================== 2. wandb 初始化（简化） ====================
wandb.init(project="raw-denoise", name="unet_noise_K4_noBN")

# ==================== 3. 数据集 ====================
def add_raw_noise(img, k=1.0, black_level=9.25 / 255.0):
    brightness = np.maximum(img - black_level, 0)
    sigma = k * np.sqrt(brightness)
    noise = np.random.normal(0, sigma, size=img.shape)
    noisy_img = img + noise
    noisy_img = np.clip(noisy_img, 0, 1)
    return noisy_img

class DenoisingDataset(Dataset):
    def __init__(self, data_dir, k_range=(0.025, 0.1)):
        self.data_dir = data_dir
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]
        self.k_range = k_range

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.image_files[idx])
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        img = img.astype(np.float32) / 65535.0

        k = random.uniform(self.k_range[0], self.k_range[1])
        noisy_img = add_raw_noise(img, k=k)

        clean_tensor = torch.from_numpy(img).float().unsqueeze(0)
        noisy_tensor = torch.from_numpy(noisy_img).float().unsqueeze(0)
        return noisy_tensor, clean_tensor

# ==================== 4. UNet (K=4, No BN) ====================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, K=4):
        super().__init__()
        c = [64//K, 128//K, 256//K, 512//K, 1024//K]
        self.down1 = DoubleConv(in_channels, c[0])
        self.down2 = DoubleConv(c[0], c[1])
        self.down3 = DoubleConv(c[1], c[2])
        self.down4 = DoubleConv(c[2], c[3])
        self.bottleneck = DoubleConv(c[3], c[4])

        self.up4 = nn.ConvTranspose2d(c[4], c[3], 2, stride=2)
        self.conv4 = DoubleConv(c[3]*2, c[3])
        self.up3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
        self.conv3 = DoubleConv(c[2]*2, c[2])
        self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
        self.conv2 = DoubleConv(c[1]*2, c[1])
        self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.conv1 = DoubleConv(c[0]*2, c[0])

        self.final = nn.Conv2d(c[0], out_channels, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))
        b = self.bottleneck(self.pool(d4))

        u4 = self.up4(b); u4 = torch.cat([u4, d4], dim=1); u4 = self.conv4(u4)
        u3 = self.up3(u4); u3 = torch.cat([u3, d3], dim=1); u3 = self.conv3(u3)
        u2 = self.up2(u3); u2 = torch.cat([u2, d2], dim=1); u2 = self.conv2(u2)
        u1 = self.up1(u2); u1 = torch.cat([u1, d1], dim=1); u1 = self.conv1(u1)
        return self.final(u1)

# ==================== 5. 保存图像函数（每10秒一次） ====================
last_save_time = 0.0
SAVE_INTERVAL = 10.0
os.makedirs("debug", exist_ok=True)  # 创建 debug 文件夹

def save_images(epoch, batch_idx, noisy_img, denoised_img, clean_img):
    global last_save_time
    now = time.time()
    if now - last_save_time < SAVE_INTERVAL:
        return False
    last_save_time = now

    def process(img):
        img = img * 255
        img = img - 9.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BAYER_RG2BGR)  # RGGB
        b, g, r = cv2.split(img)
        avg = (np.mean(r) + np.mean(g) + np.mean(b)) / 3
        r = np.clip(r * (avg / np.mean(r)), 0, 255) if np.mean(r) > 0 else r
        g = np.clip(g * (avg / np.mean(g)), 0, 255) if np.mean(g) > 0 else g
        b = np.clip(b * (avg / np.mean(b)), 0, 255) if np.mean(b) > 0 else b
        img = cv2.merge([b, g, r])
        img = (img.astype(np.float32) / 255) ** (1/2.2)
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return img

    noisy_img = process(noisy_img)
    denoised_img = process(denoised_img)
    clean_img = process(clean_img)

    # 保存到 debug 文件夹
    cv2.imwrite(f"debug/epoch_{epoch}_batch_{batch_idx}_noisy.png", noisy_img)
    cv2.imwrite(f"debug/epoch_{epoch}_batch_{batch_idx}_denoised.png", denoised_img)
    cv2.imwrite(f"debug/epoch_{epoch}_batch_{batch_idx}_clean.png", clean_img)

    print(f"Saved debug images for epoch {epoch}, batch {batch_idx+1}")
    return False  # 无需中断

# ==================== 6. 训练循环（wandb log） ====================
def train_model(model, loader, criterion, optimizer, epochs, device='cuda'):
    model.to(device)
    model.train()
    global_step = 0

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch_idx, (noisy, clean) in enumerate(loader):
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()

            pred_noise = model(noisy)
            denoised = noisy - pred_noise
            loss = criterion(denoised, clean)

            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss
            global_step += 1

            # wandb log batch loss
            wandb.log({
                "batch_loss": batch_loss,
                "global_step": global_step
            })

            print(f"Epoch [{epoch}/{epochs}], Batch [{batch_idx+1}/{len(loader)}], "
                  f"Loss: {batch_loss:.6f}")

            # 每10秒保存一次图像
            with torch.no_grad():
                n_img = noisy[0].cpu().numpy().squeeze() * 4
                d_img = denoised[0].cpu().numpy().squeeze() * 4
                c_img = clean[0].cpu().numpy().squeeze() * 4
                if save_images(epoch, batch_idx, n_img, d_img, c_img):
                    print("Training interrupted by user.")
                    return

        avg_loss = epoch_loss / len(loader)
        # wandb log epoch loss
        wandb.log({
            "epoch": epoch,
            "epoch_loss": avg_loss
        })
        print(f"Epoch [{epoch}/{epochs}] Average Loss: {avg_loss:.6f}")

# ==================== 7. 主程序 ====================
if __name__ == "__main__":
    data_dir = "data"
    batch_size = 16  # 改为16
    num_epochs = 20

    dataset = DenoisingDataset(data_dir, k_range=(0.025, 0.1))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = UNet(in_channels=1, out_channels=1, K=4)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    train_model(model, loader, criterion, optimizer, num_epochs, device)

    torch.save(model.state_dict(), "denoising_unet.pth")
    print("Training complete. Model saved as 'denoising_unet.pth'")
    wandb.finish()