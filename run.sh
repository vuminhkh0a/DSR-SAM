cd ~
cd KhoaVM
source khoa-env/bin/activate
cd y_DG
LOG_FILE="log.txt"

# ========== SL (Supervised Learning baseline) ==========
# nohup python3 -m SL.main > "$LOG_FILE" 2>&1 &

# ========== DG DualNormalization ==========
# nohup python3 -m DG.DualNormalization.main > "$LOG_FILE" 2>&1 &

# ========== DG AADG (Automatic Augmentation for Domain Generalization) ==========
# nohup python3 -m DG.AADG.main > "$LOG_FILE" 2>&1 &

# ========== DG MI-SegNet (Mutual Information Based Segmentation) ==========
# nohup python3 -m DG.MI_SegNet.main > "$LOG_FILE" 2>&1 &

# ========== DG DoFE (Domain-oriented Feature Embedding) ==========
# nohup python3 -m DG.Dofe.main > "$LOG_FILE" 2>&1 &

# ========== DG CDDSA (Contrastive Domain Disentanglement and Style Augmentation) ==========
# nohup python3 -m DG.CDDSA.main > "$LOG_FILE" 2>&1 &

# ========== DG FD (Frequency Dropout: Feature-Level Regularization via Randomized Filtering) ==========
# nohup python3 -m DG.FrequencyDropout.main > "$LOG_FILE" 2>&1 &

# ========== DG MaxStyle (Adversarial Style Composition for Robust Medical Image Segmentation) ==========
# nohup python3 -m DG.MaxStyle.main > "$LOG_FILE" 2>&1 &

# ========== DG DeSAM (Decoupled Segment Anything Model for Generalizable Medical Image Segmentation) ==========
# nohup python3 -m DG.DeSAM.main > "$LOG_FILE" 2>&1 &

# ========== DG MA-SAM (Modality-agnostic SAM Adaptation for 3D Medical Image Segmentation) ==========
# nohup python3 -m DG.MASAM.main > "$LOG_FILE" 2>&1 &

# ========== DG SR-SAM (Subspace Regularization for Domain Generalization of SAM) ==========
# nohup python3 -m DG.SR_SAM.main > "$LOG_FILE" 2>&1 &

# ========== DG SAMMed (Leveraging SAM for Single-Source DG in Medical Image Segmentation) ==========
nohup python3 -m DG.SAMMed.main > "$LOG_FILE" 2>&1 &


echo "Tailing $LOG_FILE"
tail -f "$LOG_FILE"
