# A40 拉取和运行说明

本文档用于在另一张 A40 上拉取最新版实验代码，并在你指定的数据集目录上运行或复现实验。

## 口径

当前 `train_gpu` 严格只表示 GPU 训练计算时间：

- 不包含 H2D/D2H 数据传输
- 不包含 GPU memset/init
- 不包含 objective/fit 计算
- 不包含 train/val/test metric evaluation
- 不包含 CSV 解析、日志写入、Python 数据加载等 CPU 侧时间

严格计时代码从提交 `7002c10` 开始。部署前先拉取最新 `main`。

## 1. 拉取最新版

```bash
mkdir -p /data/project/lianghan/work
cd /data/project/lianghan/work
git clone https://github.com/lianghan224-cloud/repos.git repos
cd repos
git pull
git log --oneline -3
```

如果仓库是私有的，改用你有权限的 SSH 或 token 拉取方式。

## 2. 准备数据

数据目录需要是已划分好的 split 结构：

```bash
DATA_ROOT/
  DATASET_A/
    DATASET_A_metadata.json
    DATASET_A_train.npz
    DATASET_A_val.npz
    DATASET_A_test.npz
  DATASET_B/
    ...
```

`run_newlog2_common_splits.sh` 默认数据集列表是：

```bash
DARPA LANL2 BJTaxi tpdata
```

如果你在另一张卡上使用不同数据集或只跑其中一部分，用 `DATASETS` 指定：

```bash
DATASETS="tpdata"
DATASETS="DARPA LANL2 BJTaxi tpdata"
```

如果跑 `CUTC-sgd` 或 `BLCO`，脚本需要 TNS 文件。默认会从 `.npz` 自动导出到：

```bash
${DATA_ROOT}_tns
```

也可以预先准备 TNS 目录，并通过 `TNS_ROOT=/path/to/tns` 指定。

## 3. 构建

确认 CUDA 可用：

```bash
nvidia-smi
nvcc --version
```

`run_newlog2_common_splits.sh` 会在相关方法开启时自动构建 `cuTC` 和 `BLCO`。

GenTen 需要先构建出 CUDA 版二进制：

```bash
cd /data/project/lianghan/work/repos/GenTen
mkdir -p build/cuda
cd build/cuda
cmake ../.. -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_CUDA=ON
make -j
```

确认文件存在：

```bash
ls -lh /data/project/lianghan/work/repos/GenTen/build/cuda/bin/genten
```

如果目标机已有可用 `genten`，也可以通过 `GENTEN_BIN=/path/to/genten` 指定。

## 4. 旧 CSV 快照

`artifacts/newlog2_csv_snapshot` 是旧 `train_gpu` 口径的 CSV 快照，不适合混入严格计时的新结果目录。默认不要运行：

```bash
./restore_newlog2_csv_snapshot.sh
```

该脚本默认会拒绝恢复旧快照。只有做旧口径归档对比时才允许：

```bash
ALLOW_OLD_TIMING_RESTORE=1 OUT_ROOT=/path/to/oldlog ./restore_newlog2_csv_snapshot.sh
```

## 5. 直接运行

先 dry-run 检查配置：

```bash
cd /data/project/lianghan/work/repos
ROOT=/path/to/prepared_common_splits \
OUT_ROOT=/path/to/newlog2 \
RUN_CUTC_CCD=0 RUN_CUTC_ALS=0 \
DRY_RUN=1 \
./run_newlog2_common_splits.sh
```

正式运行：

```bash
cd /data/project/lianghan/work/repos
ROOT=/path/to/prepared_common_splits \
OUT_ROOT=/path/to/newlog2 \
RUN_CUTC_CCD=0 RUN_CUTC_ALS=0 \
./run_newlog2_common_splits.sh
```

默认方法开关：

```bash
RUN_CUTC_SGD=1
RUN_CUTC_CCD=0
RUN_CUTC_ALS=0
RUN_BLCO=1
RUN_GENTEN=1
RUN_COSTCO=1
```

脚本会跳过 `OUT_ROOT` 中已有且非空的 CSV。如果你要强制重跑某个结果，先把对应 CSV 移到归档目录，或使用新的空 `OUT_ROOT`。

## 6. 等 GPU 空闲后运行

如果希望等 A40 空闲显存达到阈值后再运行：

```bash
cd /data/project/lianghan/work/repos
mkdir -p /path/to/newlog2
setsid -f env \
  ROOT=/path/to/prepared_common_splits \
  OUT_ROOT=/path/to/newlog2 \
  RUN_CUTC_CCD=0 RUN_CUTC_ALS=0 \
  MIN_FREE_MIB=43000 INTERVAL_SEC=120 \
  ./resume_newlog2_when_gpu_free.sh \
  > /path/to/newlog2/resume_launch.log 2>&1
```

查看等待状态：

```bash
cat /path/to/newlog2/resume_wait.pid
tail -f /path/to/newlog2/resume_launch.log
```

脚本会自动写：

- `resume_wait.pid`: 等待进程 PID
- `driver.pid`: 真正开始跑 runner 后的 PID
- `driver.log`: runner 输出日志

## 7. 迁移数据脚本

如果你想从当前机器同步数据到 A40：

```bash
cd /data/project/lianghan/work/repos
./sync_newlog2_resume_artifacts.sh USER@A40_HOST
```

默认行为：

- `MODE=full`: 同步所有数据集目录
- `SYNC_TNS=1`: 同步 TNS
- `SYNC_LOGS=0`: 不同步旧 CSV

如需只同步部分数据，直接用 `rsync` 或设置脚本的 `MODE=resume` 走旧的 tpdata-only 路径。

## 8. 结果检查

查看输出 CSV：

```bash
find /path/to/newlog2 -name '*.csv' | sort
```

查看运行日志：

```bash
tail -f /path/to/newlog2/driver.log
```

如果保持默认方法开关，不应生成 `CUTC-ccd` 或 `CUTC-als` 结果，除非你显式打开对应开关。
