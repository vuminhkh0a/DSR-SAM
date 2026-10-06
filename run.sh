cd ~
cd KhoaVM
source khoa-env/bin/activate
cd y_DG
LOG_FILE="log.txt"


# ========== DG UNet (plain milesial baseline) ==========
# nohup python3 -m DG.unet.main > "$LOG_FILE" 2>&1 &

# ========== DG DualNormalization (CVPR 2022, style augmentation + dual normalization) ==========
# nohup python3 -m DG.dualnormalization.main > "$LOG_FILE" 2>&1 &

# ========== DG MI-SegNet (mutual information based segmentation) ==========
# nohup python3 -m DG.mi_segnet.main > "$LOG_FILE" 2>&1 &

# ========== DG MaxStyle (MICCAI 2022, adversarial style composition) ==========
# nohup python3 -m DG.maxstyle.main > "$LOG_FILE" 2>&1 &

# ========== DG DeSAM (MICCAI 2024, decoupled SAM for generalizable segmentation) ==========
# nohup python3 -m DG.desam.main > "$LOG_FILE" 2>&1 &

# ========== DG MA-SAM (modality-agnostic SAM adaptation) ==========
# nohup python3 -m DG.masam.main > "$LOG_FILE" 2>&1 &

# ========== DG SR-SAM (subspace regularization for DG of SAM) ==========
# nohup python3 -m DG.sr_sam.main > "$LOG_FILE" 2>&1 &

# ========== DG DSR-SAM (dual subspace regularization SAM) ==========
# nohup python3 -m DG.dsr_sam.main > "$LOG_FILE" 2>&1 &

# ========== DG DSR-SAM Ablation Study ==========
nohup python3 -m DG.dsr_sam.main2 > "$LOG_FILE" 2>&1 &

# ========== DG SAMMed (leveraging SAM for single-source DG) ==========
# nohup python3 -m DG.sammed.main > "$LOG_FILE" 2>&1 &

# ========== DG SAMUS (adapting SAM for ultrasound image segmentation) ==========
# nohup python3 -m DG.samus.main > "$LOG_FILE" 2>&1 &

# ========== DG DAPSAM (domain-adaptive prompt for SAM) ==========
# nohup python3 -m DG.dapsam.main > "$LOG_FILE" 2>&1 &

# ========== DG CoSAM (self-correcting SAM) ==========
# nohup python3 -m DG.cosam.main > "$LOG_FILE" 2>&1 &

# ========== DG MedSA (medical SAM adapter) ==========
# nohup python3 -m DG.medsa.main > "$LOG_FILE" 2>&1 &

# ========== DG BUSSAM (breast ultrasound SAM adapter) ==========
# nohup python3 -m DG.bussam.main > "$LOG_FILE" 2>&1 &

# ========== DG Trans-SAM (transfer SAM with PEFT) ==========
# nohup python3 -m DG.transsam.main > "$LOG_FILE" 2>&1 &

# ========== DG Nora (noise-robust tuning of SAM) ==========
# nohup python3 -m DG.nora.main > "$LOG_FILE" 2>&1 &

# ========== DG MedSAM (official checkpoint, test-only) ==========
# nohup python3 -m DG.medsam.main > "$LOG_FILE" 2>&1 &

# ========== DG SAM-Med2D (official checkpoint, test-only) ==========
# nohup python3 -m DG.sammed2d.main > "$LOG_FILE" 2>&1 &

# ========== DG UltraSAM (official checkpoint, test-only) ==========
# nohup python3 -m DG.ultrasam.main > "$LOG_FILE" 2>&1 &


echo "Tailing $LOG_FILE"
tail -f "$LOG_FILE"
