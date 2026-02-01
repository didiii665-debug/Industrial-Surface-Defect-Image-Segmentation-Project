# 工業表面缺陷偵測
# - 因公開資源取得較為困難，我目前是暫時採用Colab合成出的金屬材質進行測試
# - 主要使用「合成金屬紋理」+「合成缺陷」產生訓練資料，去重現出實際工業生產時的狀況
# - 得出模型本體後，因應貴公司與我的討論的五大工業缺陷，加入下列更改:
# - 類別不平衡處理、細小缺陷敏感度、光照/紋理擾動、輕量化選項、標註雜訊（模擬標註不一致）

import os, math, random
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# 一、參數設定（Config）
@dataclass
class CFG:
    img_size: int = 256
    train_samples: int = 600
    val_samples: int = 120
    batch_size: int = 6
    epochs: int = 5
    lr: float = 1e-3
    seed: int = 42
    out_dir: str = "outputs_synth_metal"

    # (一) 類別不平衡模擬:用來反映真實產線通常「良品遠多於缺陷品」的壯況，避免因良品太多，導致訓練量不足的狀況
    p_defect_train: float = 0.25   # 訓練集：25% 缺陷、75% 良品（要更極端可調 0.05，較貼近現實）
    p_defect_val: float   = 0.25   # 驗證集：同樣比例（也可以改成更貼近現實狀況）

    # (二) 損失函數（Loss）組合：調整權重減少「全部預測為背景，準確率很高」的狀況
    use_dice_loss: bool = True
    use_focal_loss: bool = True
    focal_gamma: float = 2.0
    bce_weight: float = 0.5
    dice_weight: float = 0.5
    focal_weight: float = 0.5

    # (三) 細小缺陷敏感度
    # 我選擇盡量減少下採樣深度，避免細小特徵被過度壓縮、忽略
    unet_depth: int = 2   # 使用2 = 下採樣兩次較為折衷，若使用1雖然可以保持細小缺陷，但對整體的判斷較為模糊，若使用3雖然可以看比較深，但會忽略細小刮痕、缺陷等
    base_channels: int = 24  # 控制第一層捲積層Channel，讓整體模型更輕、減少資源消耗

    # (四) 光照/紋理擾動強度：模擬實際工廠中的不均勻光罩、反光、雜訊，主要是讓模型不要把光影問題誤判為缺陷
    aug_enable: bool = True
    aug_prob: float = 0.8

    # (五) 標註不一致模擬：牽涉到像素的標註，可能會因為不同人判斷有不同的標準，因此設定一個區間，避免人類標註時的主觀誤判狀況
    label_noise_enable: bool = True
    label_noise_prob: float = 0.25

    # (六) 加權抽樣器（Weighted sampler），用途為使缺陷被抽到的比率提高，但不改變「資料本身的比例」，避免模型學成「全部都預測良品」的偷懶解
    use_weighted_sampler: bool = True

cfg = CFG()
os.makedirs(cfg.out_dir, exist_ok=True)

random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# 二、合成金屬紋理背景（Synthetic metal texture）
def make_metal_texture(size: int, seed: int) -> np.ndarray:
    """
    產生類金屬灰階紋理，輸出範圍為[0,1]
    包含：
    - 基礎雜訊 + 模糊
    - 拉出金屬線條（水平/垂直 streak）
    - 輕微光照漸層
    - 對比/亮度隨機變化
    """
    rng = np.random.default_rng(seed) #使用相同的seed產出相同的金屬紋理，實現實驗重現性及方便Debug

    # 產生基礎雜訊:模擬金屬板的紋理、材質
    base = rng.normal(0.5, 0.12, (size, size)).astype(np.float32)
    base = np.clip(base, 0, 1)

    # 轉成PIL以進行模糊處理，進行高斯模糊，讓他變成接近金屬材質的質感
    img = Image.fromarray((base * 255).astype(np.uint8)).convert("L")
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(1.2, 2.5)))

    # 拉出金屬線條:先產生一個雜訊場，將其拉伸縮放+給予一個隨機方向性，讓其更近似現實的金屬質感
    streak = rng.normal(0.0, 1.0, (size, size)).astype(np.float32)
    streak = (streak - streak.min()) / (streak.max() - streak.min() + 1e-6)
    streak_img = Image.fromarray((streak * 255).astype(np.uint8)).convert("L")

    # 隨機決定要往水平或垂直方向拉動
    if rng.random() < 0.5:
        # 水平拉動:先把寬度拉長再縮回來，形成水平拖影效果
        streak_img = streak_img.resize((size * 3, size), Image.BILINEAR) \
                               .filter(ImageFilter.GaussianBlur(radius=2.0)) \
                               .resize((size, size), Image.BILINEAR)
    else:
        # 垂直拉動:把高度拉長再縮回來
        streak_img = streak_img.resize((size, size * 3), Image.BILINEAR) \
                               .filter(ImageFilter.GaussianBlur(radius=2.0)) \
                               .resize((size, size), Image.BILINEAR)

    # 將上述要素混合：結合基底紋理+拉出的金屬線條
    streak_np = np.array(streak_img).astype(np.float32) / 255.0
    tex_np = np.array(img).astype(np.float32) / 255.0
    tex_np = np.clip(tex_np * 0.75 + streak_np * 0.25, 0, 1)

    # 輕微光照漸層（模擬亮度不均），主要為了模擬光源角度、表面反射差異，避免將光源視為瑕疵
    if rng.random() < 0.5:
        grad = np.linspace(0.92, 1.08, size, dtype=np.float32)[None, :]
    else:
        grad = np.linspace(0.92, 1.08, size, dtype=np.float32)[:, None]
    tex_np = np.clip(tex_np * grad, 0, 1)

    # 隨機調整對比與亮度（增加多樣性），讓模型學習光源的結構，而不是一個絕對的亮度
    out = Image.fromarray((tex_np * 255).astype(np.uint8)).convert("L")
    out = ImageEnhance.Contrast(out).enhance(rng.uniform(0.85, 1.25))
    out = ImageEnhance.Brightness(out).enhance(rng.uniform(0.90, 1.10))

    return (np.array(out).astype(np.float32) / 255.0).astype(np.float32)

# 三、光照/紋理擾動（Illumination / texture augmentation）

def apply_illumination_aug(img_np: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    模擬真實工廠環境中的不均勻光照、反光（glare）與 gamma 變化
    目標:降低模型把紋理/光照誤判為缺陷的風險（提升泛化）
    """
    if rng.random() > cfg.aug_prob:
        return img_np

    h, w = img_np.shape
    out = img_np.copy()

    # gamma變化，改變亮度分佈，模擬不同曝光或顯示的狀況
    gamma = rng.uniform(0.7, 1.5)
    out = np.clip(out, 0, 1) ** gamma

    # 2D漸層光照（模擬照明不均），製造出某側較亮某側較暗的狀況，使照明不均勻
    gx = np.linspace(rng.uniform(0.85, 1.15), rng.uniform(0.85, 1.15), w, dtype=np.float32)
    gy = np.linspace(rng.uniform(0.85, 1.15), rng.uniform(0.85, 1.15), h, dtype=np.float32)
    grad = gy[:, None] * gx[None, :]
    out = np.clip(out * grad, 0, 1)

    # 製造局部反光:模擬金屬/亮面材質在光源下出現的一塊「局部較亮的反光」
    if rng.random() < 0.6:
        cx, cy = rng.integers(w//4, 3*w//4), rng.integers(h//4, 3*h//4)
        rr = rng.uniform(0.15, 0.35) * min(h, w)
        yy, xx = np.mgrid[0:h, 0:w]
        blob = np.exp(-((xx-cx)**2 + (yy-cy)**2) / (2*(rr**2))).astype(np.float32)
        strength = rng.uniform(0.05, 0.20)
        out = np.clip(out + blob * strength, 0, 1)

    # 加一點高斯雜訊的輔助，模擬相機、鏡頭的顆粒感
    out = np.clip(out + rng.normal(0, 0.01, size=out.shape).astype(np.float32), 0, 1)
    return out.astype(np.float32)


# (一)缺陷遮罩繪製（Defect mask drawing）
def draw_scratch(mask: Image.Image, rng: random.Random):
    """
    刮痕的組成:由折線與隨機粗細結合而成，模擬實際工廠中的金屬刮痕
    並且這裡我透過數據調整，刻意讓1px刮痕出現更頻繁，用來測試模型對細小缺陷的敏感度
    """
    draw = ImageDraw.Draw(mask)
    size = mask.size[0]
    n_points = rng.randint(2, 6)
    x, y = rng.randint(0, size-1), rng.randint(0, size-1)
    pts = [(x, y)]
    for _ in range(n_points - 1):
        x = int(np.clip(x + rng.randint(-size//2, size//2), 0, size-1))
        y = int(np.clip(y + rng.randint(-size//2, size//2), 0, size-1))
        pts.append((x, y))

    width = rng.choice([1, 1, 2, 2, 3])
    draw.line(pts, fill=255, width=width)

def draw_crack(mask: Image.Image, rng: random.Random):
    """
    裂縫:做出較長、且較鋸齒的折線
    """
    draw = ImageDraw.Draw(mask)
    size = mask.size[0]
    x0, y0 = rng.randint(0, size-1), rng.randint(0, size-1)
    angle = rng.uniform(0, 2*math.pi)
    length = rng.randint(size//2, int(size*1.2))
    steps = rng.randint(12, 28)
    pts = []
    for i in range(steps):
        t = i / (steps - 1)
        a = angle + rng.uniform(-0.30, 0.30)
        x = int(np.clip(x0 + t * length * math.cos(a), 0, size-1))
        y = int(np.clip(y0 + t * length * math.sin(a), 0, size-1))
        pts.append((x, y))
    width = rng.choice([1, 1, 2])
    draw.line(pts, fill=255, width=width)

def draw_pits(mask: Image.Image, rng: random.Random):
    """
    金屬破洞、缺口：設計成多個小圓點
    """
    draw = ImageDraw.Draw(mask)
    size = mask.size[0]
    n = rng.randint(3, 18)
    for _ in range(n):
        r = rng.randint(2, 8)
        cx = rng.randint(r, size-1-r)
        cy = rng.randint(r, size-1-r)
        draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill=255)

def apply_defect(tex: np.ndarray, soft_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    把上述所設計出來的缺陷外觀「套用」到金屬紋理上：
    - 缺陷區稍微變暗，模擬出真實性，不單單像是貼紙（例如刮痕/裂縫/污漬）
    - 加入些微雜訊，讓其貼近真實金屬材料的缺陷
    """
    out = tex.copy()

    dark = rng.uniform(0.25, 0.70)
    out = np.clip(out - soft_mask * rng.uniform(0.15, 0.35), 0, 1)
    out[soft_mask > 0.2] *= dark

    out = np.clip(out + rng.normal(0, 0.02, size=out.shape).astype(np.float32), 0, 1)
    return out.astype(np.float32)


# 四、標註雜訊（模擬標註不一致/邊界不穩）
def simulate_label_noise(mask_img: Image.Image, rng: random.Random) -> Image.Image:
    """
    模擬不同標註者，因為自身主觀影響，可能造成的邊界判斷上的差異：
    - 隨機膨脹/侵蝕
    - 邊界模糊
    """
    if not cfg.label_noise_enable:
        return mask_img
    if rng.random() > cfg.label_noise_prob:
        return mask_img

    # 形態學效果（用PIL濾鏡近似）
    if rng.random() < 0.5:
        mask_img = mask_img.filter(ImageFilter.MaxFilter(size=3))  # 膨脹
    else:
        mask_img = mask_img.filter(ImageFilter.MinFilter(size=3))  # 侵蝕

    # 模糊邊界（標註不確定），把邊界的定義改為一個區間
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.6, 1.3)))
    return mask_img

def synthesize_sample(size: int, seed: int, want_defect: bool):
    """
    生成模擬樣本所包含的內容:
    - 金屬紋理的背景模板(透過我先前設定所產生的金屬材料)
    - 依want_defect去隨機決定是否加缺陷
    - 產生soft mask再threshold成GT
    - 套用缺陷外觀
    - 進行光照擾動（設定為可開關）
    """
    rng_np = np.random.default_rng(seed)
    rng_py = random.Random(seed)

    tex = make_metal_texture(size, seed=seed)
    mask_img = Image.new("L", (size, size), 0)

    if want_defect:
        # 偏向刮痕（細小缺陷）出現更多，用來測試模型對細小缺陷的能力
        n_scratches = rng_py.choice([1, 1, 2, 2, 3])
        n_cracks = rng_py.randint(0, 1)
        n_pits = rng_py.randint(0, 1)

        for _ in range(n_scratches):
            draw_scratch(mask_img, rng_py)
        for _ in range(n_cracks):
            draw_crack(mask_img, rng_py)
        for _ in range(n_pits):
            draw_pits(mask_img, rng_py)

        # 讓缺陷邊緣更自然一些
        mask_soft = mask_img.filter(ImageFilter.GaussianBlur(radius=rng_py.uniform(0.8, 1.8)))
    else:
        # 如果為良品時:mask為全黑，代表沒有判定到瑕疵
        mask_soft = mask_img

    # 模擬標註不一致（在soft mask上做）
    mask_soft = simulate_label_noise(mask_soft, rng_py)

    soft_np = (np.array(mask_soft).astype(np.float32) / 255.0)
    gt_np = (soft_np > 0.35).astype(np.float32)  # GT 二值化

    # 套用缺陷外觀（若為良品時，soft_np幾乎是0，因此影響很小）
    img_np = apply_defect(tex, soft_np, rng_np)

    # 光照/紋理干擾:良品也模擬出來，迫使模型不要把光照當缺陷
    if cfg.aug_enable:
        img_np = apply_illumination_aug(img_np, rng_np)

    return img_np, gt_np


# 五、Dataset（控制不平衡+做個簡單的資料品質檢查(QC)）
class SynthMetalDefectDataset(Dataset):
    """
    這個 Dataset 會先依 p_defect 產生一份 labels（每筆是否為缺陷），以便：
    - 控制缺陷比例（模擬真實工廠中良品與瑕疵品不平衡的狀況）
    - 提供WeightedRandomSampler使用（可以使用，也可以不用）
    """
    def __init__(self, n_samples, size, seed=0, p_defect=0.25):
        self.n_samples = n_samples
        self.size = size
        self.seed = seed
        self.p_defect = p_defect
        self.labels = []  # 儲存每筆是否為缺陷（給sampler/QC使用）

        rng = random.Random(seed)
        for _ in range(n_samples):
            self.labels.append(1 if rng.random() < p_defect else 0)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        want_defect = bool(self.labels[idx])
        img, mask = synthesize_sample(self.size, seed=self.seed + idx, want_defect=want_defect)

        # 簡單QC:缺陷的像素比例（方便觀察標註/資料分佈是否合理）
        defect_ratio = float(mask.mean())

        img_t = torch.from_numpy(img).unsqueeze(0)   #(1,H,W)
        mask_t = torch.from_numpy(mask).unsqueeze(0) #(1,H,W)
        meta = torch.tensor([want_defect, defect_ratio], dtype=torch.float32)
        return img_t, mask_t, meta

train_ds = SynthMetalDefectDataset(cfg.train_samples, cfg.img_size, seed=123, p_defect=cfg.p_defect_train)
val_ds   = SynthMetalDefectDataset(cfg.val_samples,   cfg.img_size, seed=999, p_defect=cfg.p_defect_val)

# 加權抽樣器: 把有缺陷的樣本「抽到的機率」提高，減少模型只學習到背景的部分，提昇讓模型可以多學習判斷缺陷的比例
if cfg.use_weighted_sampler:
    labels = np.array(train_ds.labels, dtype=np.int64)
    n0 = (labels == 0).sum()
    n1 = (labels == 1).sum()
    w0 = 1.0 / max(n0, 1)
    w1 = 1.0 / max(n1, 1)
    weights = np.where(labels == 1, w1, w0).astype(np.float32)
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler, num_workers=0)
else:
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)


# 六、可調式 U-Net（深度/通道數可調，對應前面一、(三)"缺陷敏感度/控制捲積層Channel的部分"）
class DoubleConv(nn.Module):
    """
    這邊我採用U-Net常見的DoubleConv:Conv -> BN -> ReLU 重複兩次
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class UNetFlex(nn.Module):
    """
    可調深度的 U-Net：
    depth=1: 下採樣1次（較快、較輕，細節保留較多但上下文較少）
    depth=2: 下採樣2次（目前預設為此，因為整體較平衡）
    depth=3: 下採樣3次（較重，但可能表達力更高）
    """
    def __init__(self, in_ch=1, out_ch=1, base=24, depth=2):
        super().__init__()
        assert depth in [1, 2, 3]
        self.depth = depth

        self.enc1 = DoubleConv(in_ch, base)
        self.pool = nn.MaxPool2d(2)

        if depth >= 2:
            self.enc2 = DoubleConv(base, base * 2)
        if depth >= 3:
            self.enc3 = DoubleConv(base * 2, base * 4)

        # bottleneck 通道數依深度調整
        if depth == 1:
            bott_in = base
            bott_out = base * 2
        elif depth == 2:
            bott_in = base * 2
            bott_out = base * 4
        else:
            bott_in = base * 4
            bott_out = base * 8

        self.bottleneck = DoubleConv(bott_in, bott_out)

        # 上採樣路徑（decoder）: 補回早期的細節特徵，避免只靠下採後的特徵，細節部分會消失
        if depth == 1:
            self.up1 = nn.ConvTranspose2d(bott_out, base, 2, 2)
            self.dec1 = DoubleConv(base + base, base)
            self.out = nn.Conv2d(base, out_ch, 1)

        elif depth == 2:
            self.up2 = nn.ConvTranspose2d(bott_out, base * 2, 2, 2)
            self.dec2 = DoubleConv(base * 2 + base * 2, base * 2)

            self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
            self.dec1 = DoubleConv(base + base, base)

            self.out = nn.Conv2d(base, out_ch, 1)

        else:  # depth == 3
            self.up3 = nn.ConvTranspose2d(bott_out, base * 4, 2, 2)
            self.dec3 = DoubleConv(base * 4 + base * 4, base * 4)

            self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
            self.dec2 = DoubleConv(base * 2 + base * 2, base * 2)

            self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
            self.dec1 = DoubleConv(base + base, base)

            self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        x1 = self.enc1(x)

        if self.depth == 1:
            xb = self.bottleneck(self.pool(x1))
            x = self.up1(xb)
            x = torch.cat([x, x1], dim=1)
            x = self.dec1(x)
            return self.out(x)

        x2 = self.enc2(self.pool(x1))

        if self.depth == 2:
            xb = self.bottleneck(self.pool(x2))
            x = self.up2(xb)
            x = torch.cat([x, x2], dim=1)
            x = self.dec2(x)
            x = self.up1(x)
            x = torch.cat([x, x1], dim=1)
            x = self.dec1(x)
            return self.out(x)

        x3 = self.enc3(self.pool(x2))
        xb = self.bottleneck(self.pool(x3))

        x = self.up3(xb)
        x = torch.cat([x, x3], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)

        return self.out(x)

def dice_score(prob, target, eps=1e-6):
    """
    Dice分數（越高越好），常用於segmentation，對不平衡的部分更敏感，
    用來判斷"模型預測的缺陷"與"實際標註的缺陷(GT)"兩者間，重疊的比例有多大
    """
    prob = prob.view(prob.size(0), -1)
    target = target.view(target.size(0), -1)
    inter = (prob * target).sum(dim=1)
    union = prob.sum(dim=1) + target.sum(dim=1)
    dice = (2 * inter + eps) / (union + eps)
    return dice.mean().item()


# 七、損失函數:BCE + Dice + Focal

bce_loss = nn.BCEWithLogitsLoss()

def dice_loss_from_logits(logits, target, eps=1e-6):
    """
    Dice Loss：用來解決缺陷像素比例極低的不平衡問題，重疊越好>Loss會越小
    幾乎沒抓到缺陷>Loss很大
    """
    prob = torch.sigmoid(logits)
    prob = prob.view(prob.size(0), -1)
    target = target.view(target.size(0), -1)
    inter = (prob * target).sum(dim=1)
    union = prob.sum(dim=1) + target.sum(dim=1)
    dice = (2 * inter + eps) / (union + eps)
    return (1.0 - dice).mean()

def focal_loss_from_logits(logits, target, gamma=2.0, eps=1e-6):
    """
    Focal Loss：降低容易樣本的權重，讓模型更關注困難樣本（例如細小缺陷）
    避免模型只學到容易的樣本(也就是良品的樣本)，而忽略困難的樣本(有瑕疵
    的樣本)
    """
    prob = torch.sigmoid(logits).clamp(eps, 1 - eps)
    pt = torch.where(target == 1, prob, 1 - prob)
    loss = -((1 - pt) ** gamma) * torch.log(pt)
    return loss.mean()

def total_loss(logits, target):
    """
    混合損失：BCE + Dice + Focal
    """
    loss = cfg.bce_weight * bce_loss(logits, target)
    if cfg.use_dice_loss:
        loss = loss + cfg.dice_weight * dice_loss_from_logits(logits, target)
    if cfg.use_focal_loss:
        loss = loss + cfg.focal_weight * focal_loss_from_logits(logits, target, gamma=cfg.focal_gamma)
    return loss

# 建立模型（對應前面一、(三)缺陷敏感度/控制捲積層Channel的部分，可調深度與通道）
model = UNetFlex(base=cfg.base_channels, depth=cfg.unet_depth).to(device)
opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)


# 八、訓練（Train）
for ep in range(1, cfg.epochs + 1):
    model.train()
    total = 0.0

    for img, mask, meta in train_loader:
        img, mask = img.to(device), mask.to(device)

        logits = model(img)
        loss = total_loss(logits, mask)

        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()

    # 驗證：計算 Dice + 同時觀察平均defect_ratio（只是sanity check）
    model.eval()
    dices = []
    defect_ratios = []
    with torch.no_grad():
        for img, mask, meta in val_loader:
            img, mask = img.to(device), mask.to(device)
            prob = torch.sigmoid(model(img))
            dices.append(dice_score(prob, mask))
            defect_ratios.append(meta[:, 1].mean().item())

    print(
        f"Epoch {ep}/{cfg.epochs} | "
        f"train_loss={total/len(train_loader):.4f} | "
        f"val_dice={sum(dices)/len(dices):.4f} | "
        f"val_avg_defect_ratio={sum(defect_ratios)/len(defect_ratios):.4f}"
    )


# 九、視覺化（Visualize predictions）
# 採用 Image(輸入資料) / Mask (GT) / Mask (Pred)
model.eval()
imgs, masks, meta = next(iter(val_loader))
imgs, masks = imgs.to(device), masks.to(device)

with torch.no_grad():
    prob = torch.sigmoid(model(imgs))
    pred = (prob > 0.5).float()

n_show = min(4, imgs.size(0))
plt.figure(figsize=(10, 2.6 * n_show))
for i in range(n_show):
    plt.subplot(n_show, 3, i * 3 + 1)
    plt.title("Image")
    plt.imshow(imgs[i, 0].cpu().numpy(), cmap="gray")
    plt.axis("off")

    plt.subplot(n_show, 3, i * 3 + 2)
    plt.title("Mask (GT)")
    plt.imshow(masks[i, 0].cpu().numpy(), cmap="gray")
    plt.axis("off")

    plt.subplot(n_show, 3, i * 3 + 3)
    plt.title("Mask (Pred)")
    plt.imshow(pred[i, 0].cpu().numpy(), cmap="gray")
    plt.axis("off")

plt.tight_layout()
plt.show()


# 存檔:保存部分視覺化結果
save_dir = os.path.join(cfg.out_dir, "viz_samples")
os.makedirs(save_dir, exist_ok=True)
for i in range(n_show):
    im = (imgs[i, 0].cpu().numpy() * 255).astype(np.uint8)
    gt = (masks[i, 0].cpu().numpy() * 255).astype(np.uint8)
    pr = (pred[i, 0].cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(im).save(os.path.join(save_dir, f"{i:02d}_image.png"))
    Image.fromarray(gt).save(os.path.join(save_dir, f"{i:02d}_gt.png"))
    Image.fromarray(pr).save(os.path.join(save_dir, f"{i:02d}_pred.png"))

print("Saved sample visualizations to:", save_dir)