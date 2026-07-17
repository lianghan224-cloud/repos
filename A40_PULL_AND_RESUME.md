# A40 拉取和续跑说明

本文档用于在另一张 A40 上拉取 `cuTC`、`BLCO`、`CoSTCo`、`GenTen` 实验代码，并导入当前 `newlog2` 的数据和 CSV 进度后继续补完剩余训练。

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

运行脚本默认假设仓库路径是：

```bash
/data/project/lianghan/work/repos
```

因此建议直接克隆到这个位置：

```bash
mkdir -p /data/project/lianghan/work
cd /data/project/lianghan/work
git clone https://github.com/lianghan224-cloud/repos.git repos
cd repos
```

如果仓库是私有的，改用你有权限的 SSH 或 token 拉取方式。

## 2. 从当前机器同步数据和 CSV 进度

在当前这台已经有数据和日志的机器上执行，不是在 A40 上执行：

```bash
cd /data/project/lianghan/work/repos_github
./sync_newlog2_resume_artifacts.sh USER@A40_HOST
```

把 `USER@A40_HOST` 换成另一台 A40 的 SSH 登录地址。

默认 `MODE=resume` 只同步补完剩余任务需要的内容：

- `DARPA`, `LANL2`, `BJTaxi` 的 metadata，用于让脚本读取采样率后跳过已有 CSV
- `/data/project/lianghan/work/data/prepared_common_splits/tpdata`
- `/data/project/lianghan/work/data/prepared_common_splits_tns/tpdata`
- `/data/project/lianghan/work/data/prepared_common_splits_tns/manifest.json`
- `/data/project/lianghan/work/logs/newlog2` 下已有的 `.csv`

如果要完整复现全部数据集，使用：

```bash
cd /data/project/lianghan/work/repos_github
MODE=full ./sync_newlog2_resume_artifacts.sh USER@A40_HOST
```

体积参考：

- 精简续跑：metadata 很小，`tpdata` 约 701M，`tpdata` TNS 约 1.4G
- 全量 `.npz` 划分数据：约 1.8G
- 全量 TNS：约 3.7G
- `newlog2` CSV 进度：很小

## 3. 如果只从 GitHub 恢复 CSV 快照

本仓库也带了一份当前 `newlog2` CSV 快照：

```bash
artifacts/newlog2_csv_snapshot/
```

在 A40 上可执行：

```bash
cd /data/project/lianghan/work/repos
./restore_newlog2_csv_snapshot.sh
```

这只恢复 CSV 进度，不包含数据集；数据仍需用第 2 步同步。

## 4. 构建依赖

先确认 CUDA 可用：

```bash
nvidia-smi
nvcc --version
```

`run_newlog2_common_splits.sh` 会自动构建 `cuTC` 和 `BLCO`。`GenTen` 需要先构建出：

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

如果希望等 GPU 空闲到阈值后再跑：

```bash
cd /data/project/lianghan/work/repos
MIN_FREE_MIB=43000 INTERVAL_SEC=120 ./resume_newlog2_when_gpu_free.sh
```

如果要立刻跑：

```bash
cd /data/project/lianghan/work/repos
RUN_CUTC_CCD=0 RUN_CUTC_ALS=0 CUTC_TOLERANCE=1e-4 ./run_newlog2_common_splits.sh
```

脚本会根据 `/data/project/lianghan/work/logs/newlog2` 下已有 CSV 自动跳过已完成项。导入当前 CSV 进度后，正常只会继续：

1. `GenTen tpdata`
2. `CoSTCo tpdata`

## 6. 查看结果

输出目录：

```bash
/data/project/lianghan/work/logs/newlog2
```

确认剩余两个 CSV 是否生成：

```bash
ls -lh /data/project/lianghan/work/logs/newlog2/GenTen/tpdata_r8_genten_s0p5.csv
ls -lh /data/project/lianghan/work/logs/newlog2/CoSTCo/tpdata_r8_costco_s0p5.csv
```

查看驱动日志：

```bash
tail -f /data/project/lianghan/work/logs/newlog2/driver.log
```
