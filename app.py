import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

# Define FSRCNN Model
class FSRCNN(nn.Module):
    def __init__(self, scale_factor, num_channels=1, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=5//2),
            nn.PReLU(d)
        )
        self.mid_part = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m):
            self.mid_part.extend([nn.Conv2d(s, s, kernel_size=3, padding=3//2), nn.PReLU(s)])
        self.mid_part.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = nn.ConvTranspose2d(d, num_channels, kernel_size=9, stride=scale_factor, padding=9//2,
                                            output_padding=scale_factor-1)
    
    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x

# Load model weights from Google Drive
weights_2x = 'weights/fsrcnn_x2.pth'
weights_3x = 'weights/fsrcnn_x3.pth'
weights_4x = 'weights/fsrcnn_x4.pth'

# Preprocessing functions
def convert_rgb_to_ycbcr(img):
    img = np.array(img).astype(np.float32)
    y = 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
    cb = 128. + (-37.945 * img[..., 0] - 74.494 * img[..., 1] + 112.439 * img[..., 2]) / 256.
    cr = 128. + (112.439 * img[..., 0] - 94.154 * img[..., 1] - 18.285 * img[..., 2]) / 256.
    return np.array([y, cb, cr]).transpose([1, 2, 0])

def convert_ycbcr_to_rgb(img):
    r = 298.082 * img[..., 0] / 256. + 408.583 * img[..., 2] / 256. - 222.921
    g = 298.082 * img[..., 0] / 256. - 100.291 * img[..., 1] / 256. - 208.120 * img[..., 2] / 256. + 135.576
    b = 298.082 * img[..., 0] / 256. + 516.412 * img[..., 1] / 256. - 276.836
    return np.clip(np.array([r, g, b]).transpose([1, 2, 0]), 0, 255).astype(np.uint8)

# Streamlit App UI
st.title("FSRCNN Super-Resolution App")
scale_factor = st.radio("Select Upscale Factor", (2, 3, 4))
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_ycbcr = convert_rgb_to_ycbcr(image)
    img_y = img_ycbcr[..., 0] / 255.
    h, w = img_y.shape[-2:]
    img_y = torch.tensor(img_y).unsqueeze(0).unsqueeze(0).float()
    img_y = F.interpolate(img_y, size=(h * scale_factor, w * scale_factor), mode="bicubic", align_corners=False)

    # Load corresponding model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FSRCNN(scale_factor).to(device)
    if scale_factor == 2:
        model.load_state_dict(torch.load(weights_2x, map_location=device))
    elif scale_factor == 3:
        model.load_state_dict(torch.load(weights_3x, map_location=device))
    else:
        model.load_state_dict(torch.load(weights_4x, map_location=device))
    model.eval()

    with torch.no_grad():
        img_y = img_y.to(device)
        output = model(img_y).cpu().squeeze().numpy() * 255.

    h, w = img_ycbcr.shape[:2]  # Get original image height & width
    output_resized = cv2.resize(output, (w, h), interpolation=cv2.INTER_CUBIC)  
    img_ycbcr[..., 0] = output_resized  # Assign properly

    ycbcr_img = np.stack([output_resized, img_ycbcr[..., 1], img_ycbcr[..., 2]], axis=-1)
    img_sr = cv2.cvtColor(ycbcr_img.astype(np.uint8), cv2.COLOR_YCrCb2RGB)

    st.image(image, caption="Original Image", use_container_width=True)
    st.image(img_sr, caption=f"Super-Resolved Image ({scale_factor}x)", use_container_width=True)
