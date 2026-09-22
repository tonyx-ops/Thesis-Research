# GACNet 远程 CUDA 训练

这个包把现有 GACNet 对比脚本改为可移植 CPU/CUDA 版本。没有启动训练，没有租用实例。原始仓库没有被修改。包内不含约 10.3 GiB 的数据。

## 1. 创建 GPU 环境

以 Runpod 为例：Pods → Deploy，选择 1 张 RTX 4090、On-Demand、官方 Runpod PyTorch 模板（PyTorch >= 2.3）。建议 CPU 内存至少 32 GB，volume disk 50 GB、container disk 20 GB；这只是起始配置，显存峰值尚未在 GPU 上实测。

Pod 启动后：Connect → HTTP Services → JupyterLab。官方 PyTorch 模板预配 JupyterLab。上传代码 ZIP 到 /workspace，打开 JupyterLab Terminal，执行：

```bash
cd /workspace
unzip GACNet_CUDA_remote.zip
cd /workspace/GACNet_CUDA_remote
python -m pip install -r requirements-extra.txt
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

CUDA 检查必须输出 True。CUDA 运行在服务器上，本机无需安装 CUDA。以上只安装和检查环境。

## 2. 放入数据

在 /workspace/data 放入完整的 `Position_task_with_dots_synchronised_min.npz`。这份文件约 10.3 GiB，代码 ZIP 不包含它。可以通过 JupyterLab 上传，或使用该 Pod 的 SSH over exposed TCP 连接信息从本机 scp 传输。后者需要 SSH 公钥配置和开放 TCP 端口；平台的代理 SSH 通道不一定支持 scp。

下面在**本机终端**执行；把 PORT 和 HOST 替换成 Connect 页面暴露 TCP 的端口和主机，先在服务器创建 /workspace/data：

```bash
scp -P PORT '/Users/taoxue/Desktop/Thesis Research/EEG_data/Position_task_with_dots_synchronised_min.npz' root@HOST:/workspace/data/
```

也可以在服务器直接下载 Google Drive 的单个文件，避免整文件夹打包；需要该 NPZ 的具体文件链接，不能把文件夹链接当作文件下载链接。

## 3. GPU 行为检查，不训练

在服务器的包目录执行：

```bash
python run_gacnet.py --device cuda --verify-only --out /workspace/results/gpu_check
```

这会用合成数据检查前向、梯度和批次独立性，不读真实 EEG、不做 optimizer.step。它会创建 behavior_checks.json。缺少 CUDA 时会报错，不会悄悄用 CPU 跑。

## 4. 训练修正版

以下命令才开始真实训练。使用所有 21464 个样本，公开 EEGViT 代码的 ID 划分、batch 64、2 轮对比预训练 + 15 轮回归：

```bash
nohup python -u run_gacnet.py \
  --device cuda --variant corrected \
  --data /workspace/data/Position_task_with_dots_synchronised_min.npz \
  --batch-size 64 --pretrain-epochs 2 --regression-epochs 15 \
  --out /workspace/results/gac_corrected_01 \
  > /workspace/gac_corrected_01.log 2>&1 &
```

```bash
tail -f /workspace/gac_corrected_01.log
```

关闭浏览器后 nohup 进程可以继续运行，前提是 Pod 仍在运行。脚本不会自动停止 Pod。按 Ctrl+C 退出 tail 只停止查看日志。

如果显存不足，先退出失败任务，使用新输出目录并把 batch-size 改为 32 或 16。batch 变化也会改变对比损失的正负样本组成；比较不同模型时保持相同 batch。没有开启混合精度，没有实测 4090/2070 显存峰值或保证训练时间。

## 5. 对比模型

将上面 `--variant corrected` 改为 `--variant comparison`，并使用新输出目录，可依次运行适配后的原模型和修正版。相同数据划分、batch、轮数、seed、验证集选模型规则。

- `published`：保留原模型跨样本 LSTM、回归阶段 encoder no_grad 等行为。补齐运行依赖，替换加载器和缺失的 channel/cluster 输入；为公平评估使用验证集选模型。因此它是适配基线，不是原论文精确复现。
- `joint_only`：原架构只去掉 encoder 的 no_grad，便于隔离梯度问题。
- `corrected`：修复样本混用、回归梯度、图边和无正例对比损失等，并改变时序/注意力架构。
- `ablation`：published 与 joint_only。
- `comparison`：published 与 corrected。
- `all`：依次运行三种模型。

默认 2+15 轮是实验配置，不是原 GACNet 的 100+100 轮，也不代表一定收敛。想匹配原轮数，三种模型统一设置 `--pretrain-epochs 100 --regression-epochs 100`。先看较短完整数据训练的验证曲线，再决定轮数。当前 corrected 架构和原架构参数量不同，对比不能把全部差异归因于单一 bug。

## 6. 结果与数据协议

所有 EEG 保留在 CPU RAM，只有当前 batch、模型和图张量传到 GPU。每个模型目录会保存 history.csv、best_validation.pt、result.json、test_predictions.npz；根目录保存 config.json、split_indices.npz、split_report.json、数据变换参数、训练均值基线和 summary.json。

模型按最小 validation MSE 保存，训练结束才评估 test；另做固定模型的批次重排诊断，该诊断不参与选择模型。checkpoint 存为 CPU 张量，便于跨设备读取。每次使用新的 --out，防止覆盖。

当前文件：21464 样本、177 个标签 ID。按公开 EEGViT helper 的排序 ID 划分，train/val/test 为 123/27/27 个 ID，15076/3134/3254 样本。尚不能证明 ID 对应 177 个不同的人，所以不能声称严格按人跨受试者测试。原论文的 27 人和 14706/3277/3481 样本分布不能直接等同于当前代码划分。

使用公开说明中 min 文件已有的预处理，不在此重新滤波。默认额外标准化关闭；可同时为比较模型启用 --normalize-eeg --normalize-target（参数仅从 train 估计）。统一显式选择 40 个通道，原作者通道列表尚未获得。误差保存为标签原坐标单位，没有未经确认的毫米转换。输出同时区分每坐标 RMSE、二维 RMSE 和平均欧氏距离。

## 7. 下载并释放资源

训练结束后，先打包下载结果和日志：

```bash
cd /workspace
tar -czf gac_results.tar.gz results/gac_corrected_01 gac_corrected_01.log
```

在 JupyterLab 下载 gac_results.tar.gz。确认下载成功后在 Runpod 控制台 Stop 或 Terminate。Stop 后 volume 存储仍收费；Terminate 会删除未放在 network volume 的数据。network volume 需要另行删除才停止相关存储费用。

## 官方说明

- 连接/JupyterLab：https://docs.runpod.io/pods/connect-to-a-pod
- Pod 管理和数据保留：https://docs.runpod.io/pods/manage-pods
- PyTorch CUDA：https://docs.pytorch.org/docs/stable/cuda.html

本地验证只覆盖 CPU 分支、数据划分及模型行为；没有实际 CUDA 硬件验证。GPU 使用前应先执行第 3 步。
