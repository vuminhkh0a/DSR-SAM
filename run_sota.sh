#!/usr/bin/env bash
# run_sota.sh - run ALL trainable DG methods consecutively (SOTA comparison).
# Excludes the inference-only methods (MedSAM, UltraSAM, SAM-Med2D).
# Usage (do not background individual lines; the script itself runs them in order):
#   nohup bash run_sota.sh > /dev/null 2>&1 &
cd ~
cd KhoaVM
source khoa-env/bin/activate
cd y_DG
LOG_FILE="log.txt"
: > "$LOG_FILE"


# ========== DG UNet (plain milesial baseline) ==========
python3 -m DG.unet.main >> "$LOG_FILE" 2>&1

# ========== DG DualNormalization (CVPR 2022, style augmentation + dual normalization) ==========
python3 -m DG.dualnormalization.main >> "$LOG_FILE" 2>&1

# ========== DG MI-SegNet (mutual information based segmentation) ==========
python3 -m DG.mi_segnet.main >> "$LOG_FILE" 2>&1

# ========== DG MaxStyle (MICCAI 2022, adversarial style composition) ==========
python3 -m DG.maxstyle.main >> "$LOG_FILE" 2>&1

# ========== DG DeSAM (MICCAI 2024, decoupled SAM for generalizable segmentation) ==========
python3 -m DG.desam.main >> "$LOG_FILE" 2>&1

# ========== DG MA-SAM (modality-agnostic SAM adaptation) ==========
python3 -m DG.masam.main >> "$LOG_FILE" 2>&1

# ========== DG SR-SAM (subspace regularization for DG of SAM) ==========
python3 -m DG.sr_sam.main >> "$LOG_FILE" 2>&1

# ========== DG DSR-SAM no L_tsd (ablation case: L_tsd off, separate weight) ==========
python3 -m DG.dsr_sam.main_no_ltsd >> "$LOG_FILE" 2>&1

# ========== DG DSR-SAM (dual subspace regularization SAM) ==========
python3 -m DG.dsr_sam.main >> "$LOG_FILE" 2>&1

# ========== DG SAMMed (leveraging SAM for single-source DG) ==========
python3 -m DG.sammed.main >> "$LOG_FILE" 2>&1

# ========== DG SAMUS (adapting SAM for ultrasound image segmentation) ==========
python3 -m DG.samus.main >> "$LOG_FILE" 2>&1

# ========== DG DAPSAM (domain-adaptive prompt for SAM) ==========
python3 -m DG.dapsam.main >> "$LOG_FILE" 2>&1

# ========== DG CoSAM (self-correcting SAM) ==========
python3 -m DG.cosam.main >> "$LOG_FILE" 2>&1

# ========== DG MedSA (medical SAM adapter) ==========
python3 -m DG.medsa.main >> "$LOG_FILE" 2>&1

# ========== DG BUSSAM (breast ultrasound SAM adapter) ==========
python3 -m DG.bussam.main >> "$LOG_FILE" 2>&1

# ========== DG Trans-SAM (transfer SAM with PEFT) ==========
python3 -m DG.transsam.main >> "$LOG_FILE" 2>&1

# ========== DG Nora (noise-robust tuning of SAM) ==========
python3 -m DG.nora.main >> "$LOG_FILE" 2>&1


echo "All DG SOTA methods finished. Log: $LOG_FILE"
echo "Tailing $LOG_FILE"
tail -f "$LOG_FILE"
