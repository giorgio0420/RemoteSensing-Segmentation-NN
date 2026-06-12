import torch
import torch.nn as nn
import torch.nn.functional as F
from models.lightweight_unet import LightweightUNet


def haar_detail(x):
    """
    Mappa delle ALTE FREQUENZE (bordi) via Haar DWT 2D, dependency-free.
    x: [B,C,H,W] in [0,1]. Ritorna [B,C,H,W] = somma dei moduli delle 3 sottobande di dettaglio
    (LH orizzontale, HL verticale, HH diagonale), riportate a risoluzione piena.
    """
    B, C, H, W = x.shape
    filters = [
        x.new_tensor([[0.5,  0.5], [-0.5, -0.5]]),   # LH
        x.new_tensor([[0.5, -0.5], [0.5,  -0.5]]),   # HL
        x.new_tensor([[0.5, -0.5], [-0.5,  0.5]]),   # HH
    ]
    out = 0
    for f in filters:
        w = f.view(1, 1, 2, 2).repeat(C, 1, 1, 1)          # filtro depthwise (uguale per canale)
        out = out + F.conv2d(x, w, stride=2, groups=C).abs()
    return F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)


class RSPWaveletUNet(nn.Module):
    """
    Encoder RSP pre-addestrato (INTATTO) + decoder U-Net + ramo WAVELET nel decoder.

    Perche' funziona: il patch-embed di Swin riduce SUBITO l'input di 4x, quindi il dettaglio
    piu' fine (bordi sotto i ~4 px) la rete non lo vede MAI -- nessuna skip connection lo contiene.
    Qui calcoliamo le alte frequenze wavelet dell'INPUT e le iniettiamo alla risoluzione piena del
    decoder, subito prima della testa di segmentazione: cosi' i confini (strade, edifici) tornano
    netti. Encoder RSP non toccato (pretraining preservato); ramo wavelet + testa allenati da zero.
    """
    def __init__(self, num_classes, encoder_name="tu-swin_tiny_patch4_window7_224",
                 rsp_weights_path="rsp-swin-t-ckpt.pth", img_size=224, wav_ch=24):
        super().__init__()
        self.unet = LightweightUNet(num_classes=num_classes, encoder_name=encoder_name,
                                    pretraining_mode="rsp", rsp_weights_path=rsp_weights_path)

        # Hook per catturare l'output del decoder (= input del segmentation_head)
        self._dec_out = None
        self.unet.model.segmentation_head.register_forward_pre_hook(
            lambda m, inp: setattr(self, "_dec_out", inp[0]))

        # Forward fittizio per scoprire i canali del decoder (indipendenti dalla risoluzione)
        was_training = self.unet.training
        self.unet.eval()
        with torch.no_grad():
            _ = self.unet(torch.zeros(1, 3, img_size, img_size))
            dec_ch = self._dec_out.shape[1]
        self.unet.train(was_training)

        # Ramo wavelet: elabora la mappa di dettaglio e la fonde col decoder
        self.wave = nn.Sequential(
            nn.Conv2d(3, wav_ch, 3, padding=1), nn.BatchNorm2d(wav_ch), nn.ReLU(inplace=True),
            nn.Conv2d(wav_ch, wav_ch, 3, padding=1), nn.BatchNorm2d(wav_ch), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(dec_ch + wav_ch, num_classes, kernel_size=1)
        print(f"RSPWaveletUNet pronto | dec_ch={dec_ch} + wav_ch={wav_ch} -> head {num_classes} classi")

    def forward(self, x):
        self._dec_out = None
        _ = self.unet(x)                              # popola self._dec_out via hook (output orig. scartato)
        dec = self._dec_out                           # [B, dec_ch, h, w]

        hf = haar_detail(x)                           # [B, 3, H, W] (bordi dell'input)
        hf = F.interpolate(hf, size=dec.shape[-2:], mode="bilinear", align_corners=False)
        w = self.wave(hf)

        logits = self.head(torch.cat([dec, w], dim=1))
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits
