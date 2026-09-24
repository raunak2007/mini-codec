# mini-codec

A SoundStream-style neural speech codec trained from scratch on 9 hours of LibriTTS-R, plus a study of what an
adversarial loss buys and what it costs at low bitrates. One training run covers 0.5 to 4 kbps; a matched-compute
control separates what the discriminator does from what extra training does; and three evaluation axes (spectral
distance, ASR intelligibility, voice-quality proxies) disagree with each other in instructive ways.

Total compute: about 7 GPU-hours on a Modal A10, roughly $10.

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
| `base_cont` | `base` | 25,000 | no | (0, 0) | 40 min |
| `gan_soft` | `base` | 25,000 | yes | (0.03, 0.1) | 100 min |
| `gan` | `base` | 25,000 | yes | (0.1, 0.3) | 105 min |

Batch 16 × 1 s crops, Adam, lr 3e-4 with 500-step warmup and cosine decay to 3e-5; every run, including the
fine-tunes, restarts this schedule from the top. The three fine-tunes start from the `base` checkpoint with a fresh
optimiser. `base_cont` is the matched-compute control: identical to the adversarial runs except that the discriminator
is off. The mel/wave/commitment weights are identical across runs; only the adversarial terms differ.

## Results

All evaluations on test-clean. Spectral metrics use all 3,924 clips of at least 2 s; WER and voice-quality proxies
use 500 full utterances spread across speakers (54 minutes of audio). `cont` is `base_cont`, `soft` is `gan_soft`.

**Rate-distortion** (log-mel distance, lower is better; SI-SNR in dB, higher is better)

| kbps | mel: base | cont | soft | gan | SI-SNR: base | cont | soft | gan |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.493 | 0.496 | 0.522 | 0.536 | −3.0 | −3.4 | −3.1 | −2.4 |
| 1 | 0.375 | 0.371 | 0.396 | 0.412 | 1.2 | 1.0 | 1.4 | 1.7 |
| 2 | 0.303 | 0.297 | 0.319 | 0.336 | 4.4 | 4.5 | 4.9 | 5.0 |
| 4 | 0.274 | 0.263 | 0.286 | 0.305 | 5.8 | 6.2 | 6.6 | 6.7 |

**Intelligibility** (Whisper small.en WER on the reconstruction, scored against LibriTTS transcripts with Whisper's
English normaliser; WER on the original audio is 2.7%)

| kbps | base | cont | soft | gan |
|---|---|---|---|---|
| 0.5 | 14.5% | 18.5% | 20.5% | 20.6% |
| 1 | 4.8% | 5.1% | 5.1% | 5.4% |
| 2 | 3.2% | 3.1% | 3.1% | 3.4% |
| 4 | 2.9% | 3.0% | 2.9% | 3.1% |

**Voice-quality proxies** (computed on the same voiced frames for every condition; the original audio scores CPP 0.283
and flatness 0.176)

| kbps | CPP: base | cont | soft | gan | flatness: base | cont | soft | gan | balance (dB): base | cont | soft | gan |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.147 | 0.149 | 0.206 | 0.221 | 0.245 | 0.266 | 0.210 | 0.193 | 2.43 | 2.48 | 1.90 | 1.71 |
| 1 | 0.157 | 0.161 | 0.219 | 0.232 | 0.234 | 0.239 | 0.198 | 0.182 | 1.53 | 1.52 | 1.20 | 0.97 |
| 2 | 0.164 | 0.168 | 0.227 | 0.238 | 0.227 | 0.226 | 0.188 | 0.179 | 1.00 | 0.99 | 0.64 | 0.51 |
| 4 | 0.165 | 0.170 | 0.229 | 0.240 | 0.229 | 0.225 | 0.185 | 0.177 | 0.89 | 0.80 | 0.43 | 0.36 |

- *CPP*: cepstral peak prominence on voiced frames, a standard measure of harmonic clarity; low values sound breathy or
  rough. Higher and closer to the original is better.
- *flatness*: spectral flatness of the 4 to 8 kHz band on voiced frames (1 = noise). Closer to the original is better;
  above it means the top band is filled with haze.
- *balance*: mean absolute deviation of band energy (0-1, 1-2, 2-4, 4-8 kHz) from the original, in dB. Lower is better.

**Codebook utilization** (500 test utterances; `used` is codes that occur at least once out of 1024, `perplexity` is the
exponential of the code-histogram entropy, i.e. the effective codebook size, `dup` is exact duplicate rows in the trained
codebook)

| quantizer | base: used | perplexity | dup | gan_soft: used | perplexity | dup |
|---|---|---|---|---|---|---|
| 1 | 1023 | 595 | 0 | 717 | 420 | 307 |
| 2 | 1022 | 662 | 0 | 1022 | 686 | 0 |
| 3 | 1017 | 621 | 0 | 1022 | 650 | 0 |
| 4 | 1013 | 551 | 0 | 715 | 407 | 306 |
| 5 | 706 | 340 | 305 | 711 | 343 | 306 |
| 6 | 700 | 297 | 291 | 695 | 305 | 310 |
| 7 | 631 | 239 | 324 | 664 | 262 | 309 |
| 8 | 597 | 202 | 330 | 598 | 213 | 342 |

Figures: `results/compare.png` overlays the four rate-distortion curves; `results/*_curves.png` show eval during
training and the training loss. Audio: `results/samples/` has the same test clip through every model at 1 and 4 kbps.

## Findings

1. **The mel-only codec is transparent to ASR from 2 kbps up.** WER is within 0.5 points of the 2.7% floor at 2 and
   4 kbps, still 4.8% at 1 kbps, and falls off only at 0.5 kbps. Two codebooks (20 bits per 20 ms frame) are enough for
   a speech recogniser.

2. **Spectral distance and intelligibility measure different things.** Mel distance keeps improving from 2 to 4 kbps
   (0.303 → 0.274) while WER does not move. The extra bits buy fidelity, not words.

3. **The discriminator trades spectral distance for voice quality, and the trade scales with its weight.** Against the
   matched-compute control, mel distance worsens by 5 to 9% (`soft`) and 8 to 16% (`gan`), more at higher bitrates. In
   exchange, harmonic clarity recovers from 60% of the original's CPP to 81% and 85%, high-band flatness returns to
   the original's value, and spectral balance error at 4 kbps drops from 0.8 dB to 0.4 dB. Every proxy is monotone in
   the weight at every bitrate, and none of it comes from the extra training: `base_cont` moves CPP by 0.005 at most
   and flatness by 0.004 or less from 1 kbps up. This is the standard dissociation between spectral losses and
   perceived quality: the mel loss is phase-blind and compressive above 2 kHz, so the mel-only decoder smears upper
   harmonics into noise, and a discriminator that sees the complex STFT puts them back.

4. **The 0.5 kbps intelligibility cost belongs to the fine-tune, not the discriminator, and mel distance cannot see
   it.** Without the control this looks like an adversarial artefact: both adversarial runs land at 20.5% WER at
   0.5 kbps against 14.5% for `base`. But `base_cont` reaches 18.5% with no discriminator at all, so two thirds of the
   regression is the second training phase itself, and the discriminator adds about two points, which is close to the
   scatter at that error rate. Mel distance at 0.5 kbps does not register any of it (0.493 → 0.496); SI-SNR
   (−3.0 → −3.4 dB) and high-band flatness (0.245 → 0.266) do. Continued training improves the 4 kbps path (mel
   0.274 → 0.263, SI-SNR +0.4 dB) while degrading the single-codebook path: the decoder is shared across bitrates
   through quantizer dropout, so a second training phase, restarting the learning rate at 3e-4 from a checkpoint that
   had decayed to 3e-5, can move capacity between paths. This is finding 2 from the other side: at 0.5 kbps the mel
   loss is flat while intelligibility moves by a third. From 1 kbps up, `gan_soft` matches `base_cont` within 0.1
   WER points (5.1 / 3.1 / 2.9 against 5.1 / 3.1 / 3.0), so it keeps the full voice-quality gain at no
   intelligibility cost and is the configuration to prefer if 0.5 kbps is not a target.

5. **The SI-SNR gain is mostly the discriminator's.** Continued training alone adds 0.4 dB at 4 kbps, nothing at 1
   and 2 kbps, and loses 0.4 dB at 0.5 kbps. Both adversarial runs sit a further 0.4 to 0.5 dB above `base_cont` at
   1, 2 and 4 kbps, and `gan` a full 1 dB above it at 0.5 kbps. SI-SNR is a phase-sensitive metric, and the
   feature-matching loss is computed on a discriminator that sees the complex STFT, so it is the one term in the
   adversarial runs that constrains phase beyond the waveform L1.

## Caveats

- 500 utterances is about 8,500 words; WER differences under roughly 0.3 points are within noise at the 3 to 5% level.
  At 15 to 20% the scatter is wider, so the 2-point gap between `base_cont` and the adversarial runs at 0.5 kbps is
  suggestive, not established.
- `base_cont` is the matched-compute reference for everything attributed to the discriminator; comparisons against
  `base` conflate the adversarial term with 25,000 extra steps and a learning-rate restart. Findings 3 to 5 use
  `base_cont`. Whether the restart itself causes the 0.5 kbps regression is untested; a fine-tune with peak lr 3e-5
  would settle it.
- CPP, flatness and balance are proxies computed from spectrograms, not a listening test. They agree with each other and
  with what the spectrograms show, but a MOS study would be the real measure. The audio in `results/samples/` is there
  for the reader to judge.
- Codebook utilization is incomplete in the later quantizers. Each training batch holds 800 latent frames for
  1024-entry codebooks, and `vector-quantize-pytorch`'s dead-code expiry samples replacement vectors *with replacement*
  when more codes are dead than the batch has vectors, so about 300 rows of most late codebooks are exact duplicates at
  the end of training (table above). The last four quantizers therefore carry 7.6 to 8.4 effective bits per code
  instead of 10, and the nominal 4 kbps is closer to 3.6 kbps of usable capacity. Found with `eval_usage.py`; a fix is
  proposed upstream (https://github.com/lucidrains/vector-quantize-pytorch/pull/256). Retraining with the fix, or with a batch larger than the codebook, is the obvious next
  experiment and could change the 2 versus 4 kbps comparison.
- Single seed per run, 9 hours of clean read speech, one model size. Numbers are indicative of trends, not of the
  ceiling, and the size of the fine-tune effect at 0.5 kbps (4 WER points) is one measurement.

## Reproduce

Requires a Modal account (`modal setup`). Everything else is in the scripts.

```bash
pip install torch torchaudio soundfile numpy modal vector-quantize-pytorch matplotlib
modal run stage.py::stage --split dev_clean && modal run stage.py::stage --split test_clean   # LibriTTS-R into a volume
modal run train.py::prep                                                                     # resample to 16 kHz, cache
modal run --detach train.py::train --run base --steps 50000
modal run --detach train.py::train --run base_cont --init /runs/base/last.pt --steps 25000
modal run --detach train.py::train --run gan_soft  --init /runs/base/last.pt --w-adv 0.03 --w-fm 0.1 --steps 25000
modal run --detach train.py::train --run gan       --init /runs/base/last.pt --w-adv 0.1  --w-fm 0.3 --steps 25000
modal run eval_wer.py::wer --runs base,base_cont,gan_soft,gan --n 500
modal run eval_quality.py::quality --runs base,base_cont,gan_soft,gan --n 500
modal run eval_usage.py::usage --runs base,gan_soft --n 500
modal volume get codec-runs / runs && python plot.py runs/base runs/base_cont runs/gan_soft runs/gan
```

`train.py` also runs locally (`python train.py --run debug --max_files 200 --steps 200`) on CPU or Apple MPS for
development.

## Files

- `train.py`: model, losses, discriminator, training loop, Modal entrypoints (`prep`, `train`).
- `eval_wer.py`: Whisper WER per bitrate on full utterances.
- `eval_quality.py`: CPP, high-band flatness and spectral balance per bitrate.
- `eval_usage.py`: codebook utilization per quantizer (codes used, perplexity, duplicate rows).
- `plot.py`: rate-distortion, eval-during-training and training-loss figures.
- `bench.py`, `stage.py`: GPU benchmark and dataset staging.
- `results/`: eval JSON per run, WER and quality tables, figures, audio samples.
