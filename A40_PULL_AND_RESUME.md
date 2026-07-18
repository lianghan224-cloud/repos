# A40 拉取和严格 GPU 计时重跑说明

本文档用于在另一张 A40 上拉取最新版实验代码，并用已划分好的 common splits 数据集重跑 `newlog2`。

当前最新版提交：

```bash
7002c10 Restrict GPU timing to training work
```

## 口径

当前 `train_gpu` 严格只表示 GPU 训练计算时间：

- 不包含 H2D/D2H 数据传输
- 不包含 GPU memset/init
- 不包含 objective/fit 计算
- 不包含 train/val/test metric evaluation
- 不包含 CSV 解析、日志写入、Python 数据加载等 CPU 侧时间

旧 `newlog2` CSV 和 `artifacts/newlog2_csv_snapshot` 是旧口径，不能混入当前新口径结果。新 A40 上应使用空的 `OUT_ROOT` 重新生成 CSV。

## 需要重跑的实验

按当前要求，重跑：

- `CUTC-sgd`: `DARPA`, `LANL2`, `BJTaxi`, `tpdata`
- `BLCO`: `DARPA`, `LANL2`, `BJTaxi`, `tpdata`
- `GenTen`: `DARPA`, `LANL2`, `BJTaxi`, `tpdata`
- `CoSTCo`: `DARPA`, `LANL2`, `BJTaxi`, `tpdata`

`CUTC-ccd` 和 `CUTC-als` 默认关闭，不跑。

## 1. 拉取最新版

```bash
mkdir -p /data/project/lianghan/work
cd /data/project/lianghan/work
git clone https://github.com/lianghan224-cloud/repos.git repos
cd repos
git pull
git rev-parse --short HEAD
```

确认输出至少是：

```bash
7002c10
```

如果仓库是私有的，改用你有权限的 SSH 或 token 拉取方式。

## 2. 准备数据

数据目录需要是已划分好的 split 结构：

```bash
DATA_ROOT/
  DARPA/
    DARPA_metadata.json
    DARPA_train.npz
    DARPA_val.npz
    DARPA_test.npz
  LANL2/
  BJTaxi/
  tpdata/
```

如果跑 `CUTC-sgd` 或 `BLCO`，脚本会需要 TNS 文件；默认会自动导出到：

```bash
${DATA_ROOT}_tns
```

也可以预先把 TNS 目录传过去，并通过 `TNS_ROOT=/path/to/tns` 指定。

## 3. 构建

确认 CUDA 可用：

```bash
nvidia-smi
nvcc --version
```

`run_newlog2_common_splits.sh` 会在需要时自动构建 `cuTC` 和 `BLCO`。

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

## 4. 不要恢复旧 CSV 快照

当前严格 GPU 训练时间口径下，不要运行：

```bash
./restore_newlog2_csv_snapshot.sh
```

该脚本默认会拒绝恢复旧快照。只有做旧口径归档对比时才允许：

```bash
ALLOW_OLD_TIMING_RESTORE=1 OUT_ROOT=/path/to/oldlog ./restore_newlog2_csv_snapshot.sh
```

## 5. 直接重跑

建议新建空输出目录：

```bash
mkdir -p /data/project/lianghan/work/logs/newlog2
```

先 dry-run 检查配置：

```bash
cd /data/project/lianghan/work/repos
ROOT=/path/to/prepared_common_splits \
OUT_ROOT=/data/project/lianghan/work/logs/newlog2 \
RUN_CUTC_CCD=0 RUN_CUTC_ALS=0 \
DRY_RUN=1 \
./run_newlog2_common_splits.sh
```

确认无误后正式跑：

```bash
cd /data/project/lianghan/work/repos
ROOT=/path/to/prepared_common_splits \
OUT_ROOT=/data/project/lianghan/work/logs/newlog2 \
RUN_CUTC_CCD=0 RUN_CUTC_ALS=0 \
./run_newlog2_common_splits.sh
```

## 6. 等 GPU 空闲后自动跑

如果希望等 A40 空闲显存达到阈值后再跑：

```bash
cd /data/project/lianghan/work/repos
mkdir -p /data/project/lianghan/work/logs/newlog2
setsid -f env \
  ROOT=/path/to/prepared_common_splits \
  OUT_ROOT=/data/project/lianghan/work/logs/newlog2 \
  RUN_CUTC_CCD=0 RUN_CUTC_ALS=0 \
  MIN_FREE_MIB=43000 INTERVAL_SEC=120 \
  ./resume_newlog2_when_gpu_free.sh \
  > /data/project/lianghan/work/logs/newlog2/resume_launch.log 2>&1
```

查看等待状态：

```bash
cat /data/project/lianghan/work/logs/newlog2/resume_wait.pid
tail -f /data/project/lianghan/work/logs/newlog2/resume_launch.log
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

- `MODE=full`: 同步所有 4 个数据集
- `SYNC_TNS=1`: 同步 TNS
- `SYNC_LOGS=0`: 不同步旧 CSV

不要在严格 GPU 计时重跑时设置 `SYNC_LOGS=1`，否则旧 CSV 会导致 runner 跳过应重跑的实验。

## 8. 结果检查

期望生成 16 个 CSV：

```bash
find /data/project/lianghan/work/logs/newlog2 -name '*.csv' | sort
find /data/project/lianghan/work/logs/newlog2 -name '*.csv' | wc -l
```

应包含：

```bash
CUTC-sgd/*.csv
BLCO/*.csv
GenTen/*.csv
CoSTCo/*.csv
```

不应包含 `CUTC-ccd` 或 `CUTC-als` 结果，除非你显式打开了对应开关。
