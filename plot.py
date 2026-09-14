import os
import gc
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
MODEL_TYPE = "vit_h"
SOURCE = "OTU"
EPS = 1e-8

BLOCKS = [1, 10, 20, 30]
print(f'Device: {DEVICE}')

# ---------------------------------------------------------
# TỰ ĐỘNG ĐỌC EPOCH (BỎ QUA FILE KHÔNG TỒN TẠI)
# ---------------------------------------------------------
EPOCHS_TO_CHECK = list(range(1, 161))
EPOCHS = []
lora_sds = {}

for ep in EPOCHS_TO_CHECK:
    lora_path = f"weights/sr_sam/sr_sam_{MODEL_TYPE}_experiment_{SOURCE}_epoch_{ep}.pth"
    if os.path.exists(lora_path):
        lora_sds[ep] = torch.load(lora_path, map_location="cpu", weights_only=True)
        EPOCHS.append(ep)
        print(f"Loaded {lora_path}")
        
if not EPOCHS:
    raise FileNotFoundError("Không tìm thấy bất kỳ file trọng số nào! Vui lòng kiểm tra lại đường dẫn.")

print(f"Danh sách các Epochs sẽ được vẽ: {EPOCHS}")
# ---------------------------------------------------------


def compute_matrix_self_cosine_sim(block_id, branch, matrix_type, lora_sd, target_size='r', is_ema=False):
    """
    Compute absolute cosine similarity for A or B matrix with specific target size (r or dim).
    """
    tag = "ema_" if is_ema else ""
    prefix = f"sam.image_encoder.blocks.{block_id}.attn.qkv.{tag}"

    with torch.no_grad():
        weight_key = prefix + f"linear_{matrix_type}_{branch}.weight"
        M = lora_sd[weight_key].float().to(DEVICE)
        
        # Trong PyTorch: Linear A có shape (r, dim), Linear B có shape (dim, r)
        if matrix_type == "a":
            if target_size == "r":
                # Tính tương đồng giữa các hàng của A -> r x r
                norm_M = F.normalize(M, p=2, dim=1)
                sim_tensor = norm_M @ norm_M.T
            else:
                # Tính tương đồng giữa các cột của A -> dim x dim
                norm_M = F.normalize(M, p=2, dim=0)
                sim_tensor = norm_M.T @ norm_M
        else: # matrix_type == "b"
            if target_size == "r":
                # Tính tương đồng giữa các cột của B -> r x r
                norm_M = F.normalize(M, p=2, dim=0)
                sim_tensor = norm_M.T @ norm_M
            else:
                # Tính tương đồng giữa các hàng của B -> dim x dim
                norm_M = F.normalize(M, p=2, dim=1)
                sim_tensor = norm_M @ norm_M.T

        sim_matrix = sim_tensor.abs().cpu().numpy()
        del M, norm_M, sim_tensor

    if 'cuda' in DEVICE:
        torch.cuda.empty_cache()
    gc.collect()
    return sim_matrix


# ---------------------------------------------------------
# Layout setup
# 4 blocks (r x r) * Số Epochs thực tế
# ---------------------------------------------------------
nrows = len(BLOCKS) * 2 * len(EPOCHS)
ncols = 8

# Width được giữ 32, Height linh hoạt theo số hàng (Rất dài)
fig, axes = plt.subplots(nrows, ncols, figsize=(32, 4.0 * nrows))
last_im = None

col_configs = [
    ("q", "a", False, "A_Q"),
    ("q", "b", False, "B_Q"),
    ("q", "a", True,  "A_Q_ema"),
    ("q", "b", True,  "B_Q_ema"),
    ("v", "a", False, "A_V"),
    ("v", "b", False, "B_V"),
    ("v", "a", True,  "A_V_ema"),
    ("v", "b", True,  "B_V_ema"),
]

row_idx = 0

# Duyệt qua từng Block
for b in BLOCKS:
    # Bước 1: Vẽ Epochs cho kích cỡ (r x r)
    for target_size in ['r']:
        for ep in EPOCHS:
            lora_sd = lora_sds[ep]
            
            for col_idx, (branch, mat_type, is_ema, label) in enumerate(col_configs):
                ax = axes[row_idx, col_idx]
                
                sim = compute_matrix_self_cosine_sim(b, branch, mat_type, lora_sd, target_size=target_size, is_ema=is_ema)
                
                last_im = ax.imshow(
                    sim, cmap="viridis", vmin=0.0, vmax=1.0, origin="lower", aspect="equal"
                )
                del sim
                gc.collect()

                # Thiết lập Title rõ ràng hiển thị r x r hay dim x dim
                title = f"Block {b} | Epoch {ep} | Shape: {target_size}x{target_size}"
                
                # Ghi chú loại ma trận ở đầu mỗi cột của toàn bộ Figure
                if row_idx == 0:
                    title = f"--- {label} ---\n\n{title}"
                    
                ax.set_title(title, fontsize=12, fontweight="bold")
                ax.set_xlabel("Index", fontsize=9, fontweight="bold")
                ax.set_ylabel("Index", fontsize=9, fontweight="bold")
                ax.tick_params(axis="both", which="major", labelsize=8)
            
            row_idx += 1

# Colorbar chung bên phải
fig.subplots_adjust(right=0.92, top=0.99, hspace=0.4, wspace=0.3)
cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
cbar = fig.colorbar(last_im, cax=cbar_ax)
cbar.set_label("Absolute Cosine Similarity ([0.0, 1.0])", fontsize=14, fontweight="bold")
cbar.ax.tick_params(labelsize=12)

fig.suptitle(
    f"SR-SAM ({MODEL_TYPE}) A & B Matrix Absolute Self-Similarity Across Epochs\n"
    f"Blocks {BLOCKS} | Source: {SOURCE}",
    fontsize=18,
    fontweight="bold",
    y=0.998,
)

plt.tight_layout(rect=[0, 0, 0.92, 0.995])

# =========================================================
# THÊM CODE TỰ ĐỘNG LƯU HÌNH Ở ĐÂY
# =========================================================
# 1. Tạo tên file động 
blocks_str = "-".join(map(str, BLOCKS))
ep_start, ep_end = EPOCHS[0], EPOCHS[-1]
save_filename = f"SelfSim_AB_{MODEL_TYPE}_{SOURCE}_Blocks_{blocks_str}_Ep_{ep_start}to{ep_end}.png"

# 2. Tạo thư mục 'saved_plots' 
save_dir = "saved_plots"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, save_filename)

# 3. Lưu ảnh
plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='white')
print(f"✅ Đã lưu biểu đồ thành công tại: {save_path}")
# =========================================================