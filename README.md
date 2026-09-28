# ETH3D scorer

## what is this actually doing

Basically this takes a reconstruction and tells me how wrong it is compared to ETH3D

The annoying part is that the reconstruction and the ETH3D ground truth are not sitting in the
same coordinate system, so comparing the point clouds directly would mean nothing

The scorer fixes that by

1. reading the estimated camera poses
2. finding the matching real camera poses from ETH3D
3. using those cameras to figure out one rotation scale and translation
4. applying that same transform to the reconstruction
5. comparing the transformed reconstruction to the real ETH3D point cloud

The important part is that the cloud does not get to line itself up with the ground truth

That would make the score look better than it should because a bad reconstruction would get a
second chance to fix itself

This repo is only the scorer
There is no reconstruction or fusion code in here

## building it

This is meant to build inside Ubuntu WSL

```bash
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
```

The executable ends up here

```text
~/.cache/eth3d-scorer/build/dev/eth3d-scorer
```

## running it

```bash
~/.cache/eth3d-scorer/build/dev/eth3d-scorer \
  --recon recon.ply \
  --manifest manifest.json \
  --manifest-type baseline \
  --mono-root eth3d_mono_root \
  --gt reference.ply \
  --out output.json
```

So what is the manifest

It is just the JSON file where the reconstruction saved its estimated camera poses

There are two layouts right now

- `baseline` means there is one transform in `camera_poses` for each frame
- `pairwise` means the camera poses came from `accepted_pairs` and the left camera is used

Pairwise has nothing to do with the buffering idea
It only describes how the current JSON file is laid out

The other options are

- `--save-aligned` saves the reconstruction after applying the camera alignment
- `--save-trajectory` saves the camera path in ETH3D's format
- `--max-images N` only uses the first N RGB frames, with 50 as the default
- `--all-images` uses the whole sequence instead
- `--metric-points N` limits how many points from each cloud get scored
- `--threads N` sets the number of nearest-point workers
- `--thresholds ...` changes the distances used for precision recall and F1
- `--trajectory-only` skips the point clouds and only calculates trajectory scores

Zero threads means use however many CPUs are available
One thread is mostly there so the threaded answer can be checked against the simple version

The two cloud directions run one after the other
Running both at once would just quietly use twice the requested thread count

## trajectory only

This is useful when the point clouds do not matter and only the ETH3D camera score is needed

```bash
~/.cache/eth3d-scorer/build/dev/eth3d-scorer \
  --trajectory-only \
  --manifest manifest.json \
  --manifest-type pairwise \
  --mono-root eth3d_mono_root \
  --out trajectory_scores.json \
  --save-trajectory estimated_trajectory.txt \
  --all-images
```

The saved trajectory uses

```text
timestamp tx ty tz qx qy qz qw
```

That can go straight into the official ETH3D evaluator

## where is everything

- `config` reads the command and makes sure the files actually exist
- `dataset_input` blocks off the ETH3D-specific file layout from the scoring code
- `trajectory` reads camera timestamps poses and both JSON layouts
- `eth3d_trajectory_metrics` does the same trajectory math as ETH3D
- `math` handles Sim(3), rotations, camera alignment, and point sampling
- `cloud` handles PLY files, nearest points, and cloud scores
- `scorer` connects those pieces

The point of `DatasetInput` is that ETH3D should not be baked into all of the scoring math

If another dataset gets added later it should only need to provide

- its frame timestamps
- its real camera poses
- its real point cloud

The score functions return normal C++ structs
They only turn into JSON at the end when the report gets written

That is mostly so tests can check actual numbers instead of pulling everything back out of JSON

## what do the camera scores mean

All of the distance and angle errors are better when they are smaller

- `ate_mean_m` is the average distance between each estimated camera and the real camera
- `ate_median_m` is the middle camera error after sorting them
- `ate_rmse_m` is another average but large misses hurt it more
- `rotation_error_mean_deg` is the average error in which way the cameras point
- `rotation_error_median_deg` is the middle rotation error

ATE means absolute trajectory error

It is basically asking how far the estimated camera path is from the real one after lining the two
paths up as well as possible

## what is the separate ETH3D trajectory block

`eth3d_trajectory_metrics` exists so the numbers can be compared directly to the public ETH3D
evaluator

It is separate because ETH3D lines up camera centers only
The cloud alignment in this scorer also uses camera direction

The ETH3D block reports

- ATE RMSE in centimeters
- translation drift as a percentage
- rotation drift in degrees per meter
- results for path chunks around 0.5 1.0 1.5 and 2.0 meters

`pair_count` says how many chunks were actually available at that distance

If a score is `null` and `pair_count` is zero that does not mean the error was zero
It means the camera path was not long enough to form that kind of pair

`dataset_coverage` is just the fraction of the full RGB sequence that had estimated poses

## what do the cloud scores mean

The cloud gets checked in both directions because one direction is not enough

`accuracy_recon_to_gt` asks

> for every reconstructed point how far away is the closest real point

This catches points floating where nothing should exist

`completeness_gt_to_recon` asks

> for every real point how far away is the closest reconstructed point

This catches parts of the scene that never got reconstructed

A tiny but clean point cloud can have great accuracy while missing almost the entire scene
A giant messy cloud can cover everything while also putting points all over the wrong places

That is why both directions are needed

Each direction includes

- mean
- median
- RMSE
- p90
- p95

p90 means 90 percent of the point distances are at or below that value
p95 is the same idea for 95 percent

`chamfer_mean_unsquared` averages the two directional mean distances

`chamfer_mean_squared` squares the distances first, so a few really bad points hurt the score a lot
more

For each requested distance threshold

- precision is how much of the reconstruction is close enough to something real
- recall is how much of the real scene got covered
- F1 combines both, so it only gets high when both are high

## how are the cameras lining anything up

Every transform in this code means source to target

```text
target = scale * rotation * source + translation
```

So take a source point, rotate it, scale it, then move it

The point clouds store one XYZ point per row
The normal matrix equation assumes points are columns
That is why the code multiplies by the transpose when it transforms a whole cloud

The camera rotations are camera-to-world
The third column of the rotation matrix points straight ahead along the camera's positive Z axis

## why not just use the camera centers

Because centers only tell us where the cameras are
They do not fully tell us which way the cameras point

A camera path can line up perfectly while every camera is still twisted around that path

So each camera adds two alignment points

1. the camera center
2. another point a short distance straight in front of the camera

That second point forces the alignment to care about camera direction too

The forward distance is 1/100 of the normal gap between consecutive cameras
It gets calculated separately for the estimated and real paths

So it scales with the scene instead of depending on some random fixed number

Those matching points are fed into the Sim(3) solver

Sim(3) just means one transform containing

- one rotation
- one uniform scale
- one translation

The solver also rejects mirrored answers because a reflection is not a valid camera alignment

## why is there a KD-tree

The cloud scores need the closest real point for every reconstructed point and then the opposite
direction too

Checking every point against every other point would get ridiculous on a full point cloud

The KD-tree basically keeps splitting space into smaller pieces by X then Y then Z
During a search it can skip entire pieces once they are too far away to contain a better answer

It still returns the exact closest point
This is not an approximate shortcut

The tests compare it against the dumb brute-force version across a bunch of different cloud shapes

## weird ETH3D details that have to stay weird

ETH3D only keeps an RGB frame when its timestamp sits between two ground-truth readings no more
than 1/75 second apart

So `evaluated_frames` can be smaller than the number of poses in the JSON
`dataset_frames` is kept separately so that is obvious instead of looking like poses vanished

There are also two things in the official evaluator that look wrong

1. it turns the scaled 3x3 part of the Sim(3) directly into a quaternion
2. its path-pair search does not update the best distance difference while walking forward

Both behaviors are copied exactly

Fixing either one would make the code cleaner but the scores would no longer match ETH3D

## does it actually match ETH3D

Yes

The 49-pose `plant_scene_3` trajectory gets

```text
10.401514267962 cm
```

from the official ETH3D executable and the same value here to floating-point precision

A separate full-length trajectory with known drift also matches the official ATE translation and
rotation numbers at 0.5 1.0 1.5 and 2.0 meters

## what is tested

GoogleTest gets downloaded by CMake into the build cache

The tests check

- command-line options and defaults
- bad input rows actually failing instead of silently disappearing
- the exact 1/75-second timestamp cutoff
- the ETH3D trajectory calculations
- Sim(3) recovery
- camera rotation errors
- scale getting removed before a camera transform is treated like a rotation
- repeatable point sampling
- binary PLY files with properties in a different order
- KD-tree answers against brute force
- threaded answers against one thread
- the final score calculations and JSON names

