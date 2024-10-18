import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2



class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        self.enc_conv1 = self.conv_block(in_channels, 64)
        self.enc_conv2 = self.conv_block(64, 128)
        self.enc_conv3 = self.conv_block(128, 256)
        self.enc_conv4 = self.conv_block(256, 512)
        
        self.bottleneck = self.conv_block(512, 1024)

        self.up_conv4 = self.up_conv(1024, 512)
        self.dec_conv4 = self.conv_block(1024, 512)
        
        self.up_conv3 = self.up_conv(512, 256)
        self.dec_conv3 = self.conv_block(512, 256)
        
        self.up_conv2 = self.up_conv(256, 128)
        self.dec_conv2 = self.conv_block(256, 128)
        
        self.up_conv1 = self.up_conv(128, 64)
        self.dec_conv1 = self.conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def up_conv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def crop_or_pad(self, source, target):
        """Hàm để crop hoặc pad source sao cho kích thước khớp với target."""
        diff_y = target.size(2) - source.size(2)
        diff_x = target.size(3) - source.size(3)
        
        source = F.pad(source, (diff_x // 2, diff_x - diff_x // 2,
                                diff_y // 2, diff_y - diff_y // 2))
        return source    


    def forward(self, x):
        # Encoder
        enc1 = self.enc_conv1(x)                  # [320, 180]
        enc2 = self.enc_conv2(F.max_pool2d(enc1, (2, 2)))   # [160, 90]
        enc3 = self.enc_conv3(F.max_pool2d(enc2, (2, 2)))   # [80, 45]
        enc4 = self.enc_conv4(F.max_pool2d(enc3, (2, 2)))   # [40, 22]

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, (2, 2)))      # [20, 11]

        # Decoder
        dec4 = self.up_conv4(bottleneck)                              # [40, 22]
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec_conv4(dec4)
        
        dec3 = self.up_conv3(dec4)                                    # [80, 45]
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec_conv3(dec3)
        
        dec2 = self.up_conv2(dec3)                                    # [160, 90]
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec_conv2(dec2)
        
        dec1 = self.up_conv1(dec2)                                    # [320, 180]
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec_conv1(dec1)
        
        return self.final_conv(dec1)



# Dataset class
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_names = os.listdir(image_dir)
        
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        image = Image.open(os.path.join(self.image_dir, img_name))
        mask = Image.open(os.path.join(self.mask_dir, img_name)).convert('RGB')  # Convert mask to grayscale
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        # Ensure mask is binary (0 or 1)
        mask = (mask > 0).float()  # Convert mask to 0 and 1
        
        return image, mask

# Training function
def train_model(model, dataloader, criterion, optimizer, num_epochs=25, device='cuda'):
    model = model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, masks in dataloader:
            inputs = inputs.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    
    return model

def predict_and_show(model, image_path, transform, device='cuda'):
    image = Image.open(image_path)
    input_image = transform(image).unsqueeze(0).to(device)
    
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        output = model(input_image)
    
    # Tạo pred_mask
    pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
    pred_mask = (pred_mask > 0.5).astype(float)
    
    # In ra kích thước của ảnh dự đoán
    print(f"Shape of input: {input_image.shape}")
    print(f"Shape of predicted mask: {pred_mask.shape}")
    
    # Average across the channels to create a single-channel mask
    pred_mask = np.mean(pred_mask, axis=0)
    
    # Chuyển đổi mask thành dạng uint8 để dùng với OpenCV
    pred_mask_uint8 = (pred_mask * 255).astype(np.uint8)
    
    # Tạo kernel cho phép toán mở (morphological opening)
    kernel = np.ones((5, 5), np.uint8)
    
    # Áp dụng phép mở (opening) trên pred_mask
    opened_mask = cv2.morphologyEx(pred_mask_uint8, cv2.MORPH_OPEN, kernel)
    
    # Chuyển đổi opened_mask thành ảnh RGB (màu xanh lá cho vùng dự đoán)
    pred_mask_rgb = np.zeros((opened_mask.shape[0], opened_mask.shape[1], 3), dtype=np.uint8)
    pred_mask_rgb[:, :, 1] = opened_mask  # Kênh màu xanh lá
    
    # Hiển thị ảnh gốc và ảnh sau khi mở
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(pred_mask_rgb)
    ax[1].set_title('Predicted Segmentation After Opening (Green)')
    ax[1].axis('off')

    plt.show()

    # Lưu ảnh đã qua xử lý mở ra file
    output_image = Image.fromarray(pred_mask_rgb)
    output_image.save('opened_pred_mask.png')
    print("Saved opened predicted mask as 'opened_pred_mask.png'.")
transform = transforms.Compose([
    transforms.Resize((176, 320)),
    transforms.ToTensor()
])
