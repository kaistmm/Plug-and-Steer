# Face detector

S3FD face detector compatible with the face-alignment / Wav2Lip checkpoint
`s3fd-619a316812.pth` (layer names like `conv1_1`, `fc6`, ...).

Place the weight file at:

```
detectors/s3fd/weights/checkpoints/s3fd-619a316812.pth
```

Example download:

```
hf download camenduru/Wav2Lip checkpoints/s3fd-619a316812.pth \
  --local-dir detectors/s3fd/weights/
```
