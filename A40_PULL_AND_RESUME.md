# A40 拉取和续跑说明

本文档用于在另一张 A40 上拉取 `cuTC`、`BLCO`、`CoSTCo`、`GenTen` 实验代码，并在你指定的数据集路径上继续补完 `newlog2` 剩余训练。

## 当前进度

当前 `newlog2` 已完成：

- `CUTC-sgd`: `DARPA`, `LANL2`, `BJTaxi`, `tpdata`
- `BLCO`: `DARPA`, `LANL2`, `BJTaxi`, `tpdata`
- `GenTen`: `DARPA`, `LANL2`, `BJTaxi`
- `CoSTCo`: `DARPA`, `LANL2`, `BJTaxi`

当前还缺：

- `GenTen/tpdata_r8_genten_s0p5.csv`
- `CoSTCo/tpdata_r8_costco_s0p5.csv`

`cuTC` 的 `ccd` 和 `als` 当前仍按之前要求关闭，不会跑。

## 1. 在 A40 上拉代码

建议克隆到下面位置；如果克隆到其他位置也可以，脚本会自动按自身所在目录找代码：

```bash
mkdir -p /data/project/lianghan/work
cd /data/project/lianghan/work
git clone https://github.com/lianghan224-cloud/repos.git repos
cd repos
```

如果仓库是私有的，改用你有权限的 SSH 或 token 拉取方式。

## 2. 准备数据和 CSV 进度

你可以自行把数据传到 A40 任意目录。目录结构需要保持为：

```bash
DATA_ROOT/
  tpdata/
    tpdata_metadata.json
    tpdata_train.npz
    tpdata_val.npz
    tpdata_test.npz
```

如果只补当前剩余任务，只需要 `tpdata` 的上述文件，因为剩余任务是 `GenTen tpdata` 和 `CoSTCo tpdata`，不需要 TNS 文件。

本仓库带了一份当前 `newlog2` CSV 快照：

```bash
artifacts/newlog2_csv_snapshot/
```

在 A40 上恢复 CSV 进度：

```bash
cd /data/project/lianghan/work/repos
OUT_ROOT=/path/to/newlog2 ./restore_newlog2_csv_snapshot.sh
```

如果不恢复 CSV 快照，也可以直接用开关只跑剩余方法，见第 5 节。

## 3. 常用路径变量

运行脚本支持这些环境变量：

- `ROOT` 或 `DATA_ROOT`: 已划分数据集目录，例如 `/mnt/data/prepared_common_splits`
- `TNS_ROOT`: TNS 输出/读取目录，默认是 `${ROOT}_tns`
- `OUT_ROOT`: CSV 和运行日志目录，默认 `/data/project/lianghan/work/logs/newlog2`
- `WORK_TMP`: 中间文件目录
- `DATASETS`: 要跑的数据集，空格或逗号分隔，例如 `tpdata` 或 `DARPA,LANL2,BJTaxi,tpdata`
- `GENTEN_BIN`: `genten` 二进制路径
- `GPU_DEVICE`: BLCO 使用的 GPU 编号，默认 `0`
- `DRY_RUN=1`: 只打印配置，不构建、不导出、不训练

方法开关：

- `RUN_CUTC_SGD`, `RUN_CUTC_CCD`, `RUN_CUTC_ALS`
- `RUN_BLCO`
- `RUN_GENTEN`
- `RUN_COSTCO`
- `SKIP_BUILD=1`: 跳过 cuTC/BLCO 自动构建

## 4. 构建依赖

先确认 CUDA 可用：

```bash
nvidia-smi
nvcc --version
```

`run_newlog2_common_splits.sh` 只会在相关方法开启时自动构建 `cuTC` 和 `BLCO`。当前只补 `GenTen/CoSTCo` 时可以跳过它们。`GenTen` 需要先构建出：

```bash
/data/project/lianghan/work/repos/GenTen/build/cuda/bin/genten
```

构建命令按目标机环境调整；常用入口是：

```bash
cd /data/project/lianghan/work/repos/GenTen
mkdir -p build/cuda
cd build/cuda
cmake ../.. -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_CUDA=ON
make -j
```

如果目标机已有可用 `genten` 二进制，也可以直接放到上述路径。

## 5. 续跑剩余实验

假设你把数据放在：

```bash
/path/to/prepared_common_splits
```

并希望结果写到：

```bash
/path/to/newlog2
```

如果希望等 GPU 空闲到阈值后再跑：

```bash
cd /data/project/lianghan/work/repos
ROOT=/path/to/prepared_common_splits \
OUT_ROOT=/path/to/newlog2 \
DATASETS=tpdata \
RUN_CUTC_SGD=0 RUN_CUTC_CCD=0 RUN_CUTC_ALS=0 RUN_BLCO=0 \
RUN_GENTEN=1 RUN_COSTCO=1 \
MIN_FREE_MIB=43000 INTERVAL_SEC=120 \
./resume_newlog2_when_gpu_free.sh
```

如果要立刻跑：

```bash
cd /data/project/lianghan/work/repos
ROOT=/path/to/prepared_common_splits \
OUT_ROOT=/path/to/newlog2 \
DATASETS=tpdata \
RUN_CUTC_SGD=0 RUN_CUTC_CCD=0 RUN_CUTC_ALS=0 RUN_BLCO=0 \
RUN_GENTEN=1 RUN_COSTCO=1 \
./run_newlog2_common_splits.sh
```

这样不会跑 `cuTC`、`BLCO`，也不会导出 TNS；只会读取 `tpdata` 的 `.npz` 并继续：

1. `GenTen tpdata`
2. `CoSTCo tpdata`

正式跑之前可以先 dry-run 检查路径：

```bash
cd /data/project/lianghan/work/repos
ROOT=/path/to/prepared_common_splits \
OUT_ROOT=/path/to/newlog2 \
DATASETS=tpdata \
RUN_CUTC_SGD=0 RUN_CUTC_CCD=0 RUN_CUTC_ALS=0 RUN_BLCO=0 \
RUN_GENTEN=1 RUN_COSTCO=1 \
DRY_RUN=1 \
./run_newlog2_common_splits.sh
```

如果你已经恢复了 CSV 快照，也可以跑全量数据集，脚本会自动跳过已有 CSV：

```bash
cd /data/project/lianghan/work/repos
ROOT=/path/to/prepared_common_splits \
OUT_ROOT=/path/to/newlog2 \
RUN_CUTC_CCD=0 RUN_CUTC_ALS=0 \
./run_newlog2_common_splits.sh
```

## 6. 查看结果

输出目录由 `OUT_ROOT` 指定，例如：

```bash
/path/to/newlog2
```

确认剩余两个 CSV 是否生成：

```bash
ls -lh /path/to/newlog2/GenTen/tpdata_r8_genten_s0p5.csv
ls -lh /path/to/newlog2/CoSTCo/tpdata_r8_costco_s0p5.csv
```

查看驱动日志：

```bash
tail -f /path/to/newlog2/driver.log
```
