import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def _up_block(in_ch, out_ch):
    """Upsample x2 + conv: ricostruisce risoluzione gradualmente (maschere meno sfocate)."""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class SatMAEppSegmenter(nn.Module):
    """
    STADIO 1 (RGB) — SatMAE++ come encoder ViT-L *CONGELATO* + decoder leggero.

    Perche' cosi' gira su Colab: congelando l'encoder NON si memorizzano attivazioni per il
    backward -> memoria e tempo crollano, ViT-L diventa fattibile su T4. Si allena solo il decoder.
    In data-scarce (100-300 img) un run dura pochi minuti, niente nottate.

    Requisiti:
      - `timm` (scheletro ViT-L/16 @224)
      - checkpoint SatMAE++ ViT-L fMoW-RGB in `ckpt_path` (repo: github.com/techmn/satmae_pp)

    NOTA ONESTA: la rimappatura delle chiavi puo' richiedere un ritocco a seconda di come e'
    salvato il file. Il modulo STAMPA quante chiavi combaciano: se il match e' basso, gira prima
    lo SMOKE TEST (sotto) e mi mandi l'output, sistemo la mappatura come abbiamo fatto con RSP.
    """

    def __init__(self, num_classes=8, ckpt_path="satmaepp_vitl_fmow.pth",
                 img_size=224, freeze_encoder=True, pretrained=True):
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise ImportError("Serve `timm` per SatMAE++ (pip install timm).") from e

        self.img_size = img_size
        self.patch = 16
        self.grid = img_size // self.patch      # 14
        self.embed_dim = 1024                   # ViT-L
        self.freeze_encoder = freeze_encoder

        # Scheletro ViT-L/16 (i pesi SatMAE++ vengono caricati sopra)
        self.encoder = timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=0)
        if pretrained:
            self.encoder_loaded = self._load_satmae(ckpt_path)
        else:
            self.encoder_loaded = False
            print("[SatMAE++] Encoder ViT-L RANDOM (controllo: stessa rete SENZA pretraining).")

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

        # Decoder: [B,1024,14,14] -> upsample progressivo 14->28->56->112->224
        self.decoder = nn.Sequential(
            _up_block(self.embed_dim, 512),
            _up_block(512, 256),
            _up_block(256, 128),
            _up_block(128, 64),
        )
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def train(self, mode=True):
        """Tiene SEMPRE l'encoder in eval (BN/dropout fermi) anche quando alleniamo il decoder."""
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def _load_satmae(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            print(f"[SatMAE++] ⚠️ checkpoint '{ckpt_path}' non trovato: encoder RANDOM.")
            return False

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            for key in ("model", "state_dict"):
                if key in ckpt:
                    ckpt = ckpt[key]
                    break

        # Checkpoint MAE/SatMAE usano nomi compatibili con timm (patch_embed.proj, blocks.*,
        # norm, pos_embed, cls_token). Togliamo i prefissi comuni e le chiavi del decoder MAE.
        cleaned = {}
        for k, v in ckpt.items():
            nk = k.replace("module.", "").replace("encoder.", "")
            if nk.startswith("decoder") or "mask_token" in nk:
                continue
            cleaned[nk] = v

        missing, unexpected = self.encoder.load_state_dict(cleaned, strict=False)
        total = len(self.encoder.state_dict())
        matched = total - len(missing)
        print(f"[SatMAE++] Pesi caricati: {matched}/{total} tensori "
              f"(missing={len(missing)}, unexpected={len(unexpected)}).")
        if matched < 0.5 * total:
            print("[SatMAE++] ⚠️ Match basso: i nomi delle chiavi non combaciano. "
                  "Manda l'output dello SMOKE TEST e sistemo la rimappatura.")
            return False
        print("[SatMAE++] 🚀 Encoder ViT-L inizializzato con pesi SatMAE++ (congelato).")
        return True

    def _forward_tokens(self, x):
        feats = self.encoder.forward_features(x)          # timm: [B, N(+1), C]
        if feats.dim() == 3 and feats.shape[1] == self.grid * self.grid + 1:
            feats = feats[:, 1:, :]                        # rimuovi cls token
        return feats

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        with torch.no_grad():
            feats = self._forward_tokens(x)               # [B, 196, 1024]
        B = x.shape[0]
        feats = feats.transpose(1, 2).reshape(B, self.embed_dim, self.grid, self.grid)  # [B,1024,14,14]
        d = self.decoder(feats)                           # [B,64,224,224]
        if d.shape[-2:] != (H, W):
            d = F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False)
        return self.classifier(d)
