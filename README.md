# mini-codec

A SoundStream-style neural speech codec trained from scratch on 9 hours of LibriTTS-R, plus a study of what an
adversarial loss buys and what it costs at low bitrates. One training run covers 0.5 to 4 kbps; three evaluation
axes (spectral distance, ASR intelligibility, voice-quality proxies) disagree with each other in instructive ways.

Total compute: about 5 GPU-hours on a Modal A10, roughly $8.

## Model

- **Encoder / decoder**: 1-D convolutional, strides (2, 4, 5, 8) for a hop of 320 samples, i.e. 50 latent frames per
  second at 16 kHz. Residual units with dilations 1, 3, 9. 9.84M parameters, 128-d latent.
- **Quantizer**: residual vector quantization, 8 codebooks × 1024 entries (10 bits per frame per codebook), so each
  codebook adds 0.5 kbps: 0.5, 1, 2, 4 kbps with 1, 2, 4, 8 codebooks. EMA codebooks with k-means initialisation
  (`vector-quantize-pytorch`).
- **Quantizer dropout**: each training batch decodes from a uniformly random number of codebooks (1 to 8), so a single
  model serves the whole bitrate range and the rate-distortion curve comes from one run.
- **Reconstruction loss**: log-mel L1 at three STFT resolutions (512 / 1024 / 2048, 64 mels) + waveform L1 + commitment.
- **Adversarial loss (optional)**: 4-scale STFT discriminator (n_fft 1024 / 512 / 256 / 128; 2-D convs over the complex
  spectrogram, weight-normalised, 0.45M parameters), hinge loss plus feature matching. EnCodec's discriminator design
  with HiFi-GAN / DAC-style loss weighting.

## Data

LibriTTS-R, resampled 24 → 16 kHz. Train on `dev-clean` (5,736 utterances, 40 speakers, 8.93 h), evaluate on
`test-clean` (4,837 utterances, 39 speakers, 8.54 h). The splits are speaker-disjoint, so every number below is on
unseen speakers.

## Runs

| run | init | steps | discriminator | (w_adv, w_fm) | A10 time |
|---|---|---|---|---|---|
| `base` | scratch | 50,000 | no | (0, 0) | 75 min |
| `gan_soft` | `base` | 25,000 | yes | (0.03, 0.1) | 100 min |
| `gan` | `base` | 25,000 | yes | (0.1, 0.3) | 105 min |

Batch 16 × 1 s crops, Adam, lr 3e-4 with 500-step warmup and cosine decay to 3e-5. The adversarial runs are
fine-tunes from the `base` checkpoint with a fresh discriminator. The mel/wave/commitment weights are identical
across runs; only the adversarial terms differ.

## Results

All evaluations on test-clean. Spectral metrics use all 3,924 clips of at least 2 s; WER and voice-quality proxies
use 500 full utterances spread across speakers (54 minutes of audio).

**Rate-distortion** (log-mel distance, lower is better; SI-SNR in dB, higher is better)

| kbps | mel: base | soft | gan | SI-SNR: base | soft | gan |
|---|---|---|---|---|---|---|
| 0.5 | 0.493 | 0.522 | 0.536 | −3.0 | −3.1 | −2.4 |
| 1 | 0.375 | 0.396 | 0.412 | 1.2 | 1.4 | 1.7 |
| 2 | 0.303 | 0.319 | 0.336 | 4.4 | 4.9 | 5.0 |
| 4 | 0.274 | 0.286 | 0.305 | 5.8 | 6.6 | 6.7 |

**Intelligibility** (Whisper small.en WER on the reconstruction, scored against LibriTTS transcripts with Whisper's
English normaliser; WER on the original audio is 2.7%)

| kbps | base | soft | gan |
|---|---|---|---|
| 0.5 | 14.5% | 20.5% | 20.6% |
| 1 | 4.8% | 5.1% | 5.4% |
| 2 | 3.2% | 3.1% | 3.4% |
| 4 | 2.9% | 2.9% | 3.1% |

**Voice-quality proxies** (computed on the same voiced frames for every condition; the original audio scores CPP 0.283
and flatness 0.176)

| kbps | CPP: base | soft | gan | flatness: base | soft | gan | balance (dB): base | soft | gan |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.147 | 0.206 | 0.221 | 0.245 | 0.210 | 0.193 | 2.43 | 1.90 | 1.71 |
| 1 | 0.157 | 0.219 | 0.232 | 0.234 | 0.198 | 0.182 | 1.53 | 1.20 | 0.97 |
| 2 | 0.164 | 0.227 | 0.238 | 0.227 | 0.188 | 0.179 | 1.00 | 0.64 | 0.51 |
| 4 | 0.165 | 0.229 | 0.240 | 0.229 | 0.185 | 0.177 | 0.89 | 0.43 | 0.36 |

- *CPP*: cepstral peak prominence on voiced frames, a standard measure of harmonic clarity; low values sound breathy or
  rough. Higher and closer to the original is better.
- *flatness*: spectral flatness of the 4 to 8 kHz band on voiced frames (1 = noise). Closer to the original is better;
  above it means the top band is filled with haze.
- *balance*: mean absolute deviation of band energy (0-1, 1-2, 2-4, 4-8 kHz) from the original, in dB. Lower is better.

Figures: `results/compare.png` overlays the three rate-distortion curves; `results/*_curves.png` show eval during
training and the training loss. Audio: `results/samples/` has the same test clip through every model at 1 and 4 kbps.

## Findings

1. **The mel-only codec is transparent to ASR from 2 kbps up.** WER is within 0.5 points of the 2.7% floor at 2 and
   4 kbps, still 4.8% at 1 kbps, and falls off only at 0.5 kbps. Two codebooks (20 bits per 20 ms frame) are enough for
   a speech recogniser.

2. **Spectral distance and intelligibility measure different things.** Mel distance keeps improving from 2 to 4 kbps
   (0.303 → 0.274) while WER does not move. The extra bits buy fidelity, not words.

3. **The discriminator trades spectral distance for voice quality, and the trade scales with its weight.** Mel distance
   worsens by about 4% (`soft`) and 11% (`gan`) across the board. In exchange, harmonic clarity recovers from 58% of the
   original's CPP to 81% and 85%, high-band flatness returns to the original's value, and spectral balance error drops
   from 0.9 dB to 0.4 dB at 4 kbps. Every proxy is monotone in the weight at every bitrate. This is the standard
   dissociation between spectral losses and perceived quality: the mel loss is phase-blind and compressive above 2 kHz,
   so the mel-only decoder smears upper harmonics into noise, and a discriminator that sees the complex STFT puts them
   back.

4. **The intelligibility cost is concentrated at 0.5 kbps and is a threshold, not a dial.** Both adversarial runs land
   at 20.5% WER at 0.5 kbps against 14.5% for `base`, regardless of weight, while at 1 kbps and above `gan_soft` matches
   `base` within noise. At one codebook the code does not carry enough information to pin down the fine structure, and
   the discriminator pushes the decoder to invent speech-like texture anyway: CPP rises (0.147 → 0.206) while WER rises
   with it. The output is more speech-shaped and less correct. `gan_soft` keeps essentially all of the voice-quality
   gain at no intelligibility cost from 1 kbps up, so it is the configuration to prefer if 0.5 kbps is not a target.

5. **SI-SNR improves with the adversarial runs, but not because of the discriminator.** Both `gan_soft` and `gan` gain
   0.5 to 0.9 dB over `base` and are equal to each other, which points to the additional 25,000 steps of waveform loss
   rather than the adversarial term.

<!-- control run (base_cont: base continued 25k steps, no discriminator) goes here: does 0.5 kbps WER stay at 14.5%,
     and does SI-SNR reach 6.6 dB without a discriminator? -->

## Caveats

- 500 utterances is about 8,500 words; WER differences under roughly 0.3 points are within noise.
- The adversarial runs have 25,000 more training steps than `base`. The control run above is the right comparison for
  anything attributed to training length.
- CPP, flatness and balance are proxies computed from spectrograms, not a listening test. They agree with each other and
  with what the spectrograms show, but a MOS study would be the real measure. The audio in `results/samples/` is there
  for the reader to judge.
- Single seed, 9 hours of clean read speech, one model size. Numbers are indicative of trends, not of the ceiling.

## Reproduce

Requires a Modal account (`modal setup`). Everything else is in the scripts.

```bash
pip install torch torchaudio soundfile numpy modal vector-quantize-pytorch matplotlib
modal run stage.py::stage --split dev_clean && modal run stage.py::stage --split test_clean   # LibriTTS-R into a volume
modal run train.py::prep                                                                     # resample to 16 kHz, cache
modal run --detach train.py::train --run base --steps 50000
modal run --detach train.py::train --run gan_soft --init /runs/base/last.pt --w-adv 0.03 --w-fm 0.1 --steps 25000
modal run --detach train.py::train --run gan      --init /runs/base/last.pt --w-adv 0.1  --w-fm 0.3 --steps 25000
modal run eval_wer.py::wer --runs base,gan_soft,gan --n 500
modal run eval_quality.py::quality --runs base,gan_soft,gan --n 500
modal volume get codec-runs / runs && python plot.py runs/base runs/gan_soft runs/gan
```

`train.py` also runs locally (`python train.py --run debug --max_files 200 --steps 200`) on CPU or Apple MPS for
development.

## Files

- `train.py`: model, losses, discriminator, training loop, Modal entrypoints (`prep`, `train`).
- `eval_wer.py`: Whisper WER per bitrate on full utterances.
- `eval_quality.py`: CPP, high-band flatness and spectral balance per bitrate.
- `plot.py`: rate-distortion, eval-during-training and training-loss figures.
- `bench.py`, `stage.py`: GPU benchmark and dataset staging.
- `results/`: eval JSON, WER and quality tables, figures, audio samples.
