"""Wild VLN MVP data pipeline.

Stages (each writes an immutable artifact + QC report under
/data/patelm/ticvla/wildvln/):
  p0  inventory   - per-bag metadata manifest + rig-contract checks
  p1  odometry    - KISS-ICP poses, keyframe index, GPS/twist tracks
  p2a voxelmaps   - timestamped LiDAR voxel maps (GT only, never model input)
  p2b features    - metric depth (LiDAR-fitted) + ViT patch-feature cache
  p3  windows     - 20 s windows, distance-parameterized traces, BEV crops
  p4  language    - landmark registry, T0-T3 labels, memory chains
  p5  packaging   - webdataset shards + splits
"""
