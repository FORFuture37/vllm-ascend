# MiniMax-M3

## 1 Introduction

MiniMax-M3 is a multimodal large language model that supports text, image, and video inputs. On Ascend, it supports BF16 and W8A8 on A2/A3, MXFP8 PD disaggregation on 950DT, thinking mode, reasoning parsing, tool-call parsing, and multimodal inputs.

This document covers supported features, environment and model preparation, single-node deployment, multi-node deployment, PD separation, thinking and parser configuration, functional verification, accuracy evaluation, and troubleshooting.

This document is written based on the latest vLLM-Ascend version. This model is supported on the main branch.

## 2 Supported Features

Refer to [Supported Features List](../../user_guide/support_matrix/supported_models.md) for the model support matrix.

Refer to the [Feature Guide](../../user_guide/feature_guide/index.md) for feature configuration instructions.

## 3 Prerequisites

### 3.1 Model Weight

- `MiniMax-M3` (BF16): requires 16 × 64 GB NPU chips. [Download the model weights](https://www.modelscope.cn/collections/MiniMax/MiniMax-M3).
- `MiniMax-M3-w8a8` (W8A8): requires at least 8 × 64 GB NPU chips. Recommended for Atlas 800 A3 (64GB × 16) and Atlas 800 A2 (64GB × 8). [Download the model weights](https://www.modelscope.cn/models/Eco-Tech/MiniMax-M3-w8a8-0626).
- `MiniMax-M3-MXFP8` (MXFP8): used for Ascend 950DT (96GB × 8) PD disaggregation.

It is recommended to place the model weight in a shared cache directory.

### 3.2 Verify Multi-node Communication (Optional)

For multi-node deployment, verify the communication environment by following [Verify Multi-node Communication Environment](../../getting_started/installation.md#installation-multi-node-interconnect).

## 4 Installation

### 4.1 Docker Image Installation

You can use the official all-in-one Docker image. For the available image tags and published versions, refer to [Using Docker](../../getting_started/installation.md#installation-prebuilt-image).

=== "A3 series"

    **Docker Run:**

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a3
    export NAME=vllm-ascend

    docker run --rm \
        --name $NAME \
        --net=host \
        --shm-size=100g \
        --device /dev/davinci0 \
        --device /dev/davinci1 \
        --device /dev/davinci2 \
        --device /dev/davinci3 \
        --device /dev/davinci4 \
        --device /dev/davinci5 \
        --device /dev/davinci6 \
        --device /dev/davinci7 \
        --device /dev/davinci8 \
        --device /dev/davinci9 \
        --device /dev/davinci10 \
        --device /dev/davinci11 \
        --device /dev/davinci12 \
        --device /dev/davinci13 \
        --device /dev/davinci14 \
        --device /dev/davinci15 \
        --device /dev/davinci_manager \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
        -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        -v /root/.cache:/root/.cache \
        -it $IMAGE bash
    ```

    !!! note

        A3 has 8 NPUs with dual-die design (16 chips total: `/dev/davinci[0-15]`).
        If you are on a shared machine, map only the chips you need (e.g., `/dev/davinci[0-7]` for NPU 0-3).

=== "Ascend 950DT series"

    **Docker Run:**

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-950DT
    export NAME=vllm-ascend

    docker run --rm \
        --name $NAME \
        --net=host \
        --shm-size=100g \
        --device /dev/davinci0 \
        --device /dev/davinci1 \
        --device /dev/davinci2 \
        --device /dev/davinci3 \
        --device /dev/davinci4 \
        --device /dev/davinci5 \
        --device /dev/davinci6 \
        --device /dev/davinci7 \
        --device /dev/davinci_manager \
        --device /dev/hisi_hdc \
        --device /dev/ummu \
        --device /dev/uburma \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        -v /etc/hccl_rootinfo.json:/etc/hccl_rootinfo.json \
        -v /etc/hixlep/:/etc/hixlep/ \
        -v /root/.cache:/root/.cache \
        -v /usr/local/sbin:/usr/local/sbin \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
        -v /usr/lib64:/usr/lib64 \
        -it $IMAGE bash
    ```

    !!! note

        Ascend 950DT has 8 NPUs (`/dev/davinci[0-7]`). Mount `/etc/hixlep/` for UBOE communication when using PD disaggregation.

=== "A2 series"

    **Docker Run:**

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
    export NAME=vllm-ascend

    docker run --rm \
        --name $NAME \
        --net=host \
        --shm-size=100g \
        --device /dev/davinci0 \
        --device /dev/davinci1 \
        --device /dev/davinci2 \
        --device /dev/davinci3 \
        --device /dev/davinci4 \
        --device /dev/davinci5 \
        --device /dev/davinci6 \
        --device /dev/davinci7 \
        --device /dev/davinci_manager \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
        -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        -v /root/.cache:/root/.cache \
        -it $IMAGE bash
    ```

!!! tip
    The mounts above are the minimum required for NPU driver access. Add additional `-v` mounts (e.g., model weight paths, datasets) as needed for your environment.

The default workdir is `/workspace`. vLLM and vLLM-Ascend are installed as Python packages in site-packages.

MiniMax-M3 needs the Rust frontend parser. After the container starts, run:

```bash
cd /vllm-workspace/vllm
pip install setuptools-rust
./build_rust.sh
```

**Installation Verification:**

After starting the container, run the following command to verify the installation:

```bash
docker ps | grep vllm-ascend-env
```

Expected result: The container is listed with status `Up`. You can also verify the vllm-ascend version inside the container:

```bash
pip show vllm-ascend
```

Expected result: The version information is displayed, matching the pulled image version.

### 4.2 Source Code Installation

If you prefer to build from source instead of using the Docker image, install vLLM-Ascend following the [Installation Guide](../../getting_started/installation.md).

To verify the source installation:

```bash
python -c "import vllm_ascend; print(vllm_ascend.__version__)"
```

## 5 Online Service Deployment

Start the online serving service with the following command:

For descriptions of the standard `vllm serve` arguments used in the deployment examples, refer to the [vLLM Serving Arguments documentation](https://docs.vllm.ai/en/latest/cli/serve/#arguments). For Ascend-specific options passed through `--additional-config`, refer to [Additional Configuration](../../user_guide/configuration/additional_config.md). For Ascend-specific environment variables, refer to [Environment Variables](../../user_guide/configuration/env_vars.md).

### 5.1 Single-Node Deployment

Single-node deployment completes both Prefill and Decode within the same node. Both the bfloat and quantized model can be deployed on 1 Atlas 800 A3 (64GB × 16). Quantized model can be deployed on 1 Atlas 800 A2 (64GB × 8).

#### BF16 Deployment

```bash
export HCCL_OP_EXPANSION_MODE="AIV"
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve ${WEIGHT_PATH} \
    --served-model-name minimax-m3 \
    --trust-remote-code \
    --max-model-len 43008 \
    --tensor-parallel-size 16 \
    --enable-expert-parallel \
    --max-num-seqs 16 \
    --distributed_executor_backend "mp" \
    --gpu-memory-utilization 0.92 \
    --reasoning-parser minimax_m3 \
    --limit-mm-per-prompt '{"image":1,"video":0}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_cpu_binding":true,"ascend_compilation_config":{"enable_static_kernel":true,"fuse_norm_quant":false},"multistream_overlap_shared_expert":true,"weight_nz_mode":2,"enable_flashcomm1":true,"enable_reduce_sample":true}' \
    --port 11223 > ${LOG_PATH} 2>&1 &
```

#### W8A8 Deployment

```bash
export HCCL_OP_EXPANSION_MODE="AIV"
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve ${WEIGHT_PATH} \
    --served-model-name minimax-m3 \
    --trust-remote-code \
    --max-model-len 131072 \
    --tensor-parallel-size 4 \
    --data-parallel-size 4 --api_server_count 1 \
    --max-num-batched-tokens 32768 \
    --long-prefill-token-threshold 4096 \
    --enable-expert-parallel \
    --max-num-seqs 32 \
    --distributed_executor_backend "mp" \
    --gpu-memory-utilization 0.92 \
    --reasoning-parser minimax_m3 \
    --limit-mm-per-prompt '{"image":1,"video":0}' \
    --speculative-config '{"model":"${EAGLE3_WEIGHT_PATH}", "method":"eagle3", "num_speculative_tokens":3}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_cpu_binding":true,"ascend_compilation_config":{"enable_static_kernel":true,"fuse_norm_quant":false},"multistream_overlap_shared_expert":true,"enable_shared_expert_dp":true,"weight_nz_mode":2,"enable_flashcomm1":true,"enable_reduce_sample":true}' \
    --port 11223 > ${LOG_PATH} 2>&1 &
```

**Note**: In the script above, `max-num-seqs` is set to 16, which represents the maximum number of sequences the scheduler can process in a single iteration. Adjust the `max-num-seqs` parameter dynamically based on actual business.

For text-only deployment, `--limit-mm-per-prompt` can be omitted. For multimodal deployment, configure this parameter according to the actual request shape. For example, use `--limit-mm-per-prompt '{"image":2,"video":0}'` for two-image requests, and use `--limit-mm-per-prompt '{"image":0,"video":1}'` for one-video requests.

### 5.2 Multi-Node Deployment

Deploying the float model on Ascend A2 servers requires at least two nodes. Multi-node deployment on A3 servers without prefill–decode disaggregation is not recommended. Update `WEIGHT_PATH`, `EAGLE3_WEIGHT_PATH`, `LOG_PATH`, `local_ip`, `node0_ip`, and `IFNAME` based on the actual environment.

#### BF16 Deployment

Run the following command on node 0:

```bash
local_ip="${NODE0_IP}"
node0_ip="${NODE0_IP}"

export HCCL_IF_IP=$local_ip
export IFNAME="${NETWORK_INTERFACE}"
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_SOCKET_IFNAME="$IFNAME"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export GLOO_SOCKET_IFNAME="$IFNAME"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve ${WEIGHT_PATH} \
    --host 0.0.0.0 \
    --served-model-name minimax-m3 \
    --trust-remote-code \
    --max-model-len 40960 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --max-num-seqs 8 \
    --data-parallel-size 2 \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank 0 \
    --data-parallel-address $node0_ip \
    --distributed_executor_backend "mp" \
    --gpu-memory-utilization 0.94 \
    --reasoning-parser minimax_m3 \
    --limit-mm-per-prompt '{"image":1,"video":0}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_cpu_binding":true, "ascend_compilation_config":{"fuse_norm_quant":false}, "multistream_overlap_shared_expert": true, "weight_nz_mode": 2}' \
    --port 11223 > ${LOG_PATH} 2>&1 &
```

Run the following command on node 1:

```bash
local_ip="${NODE1_IP}"
node0_ip="${NODE0_IP}"

export HCCL_IF_IP=$local_ip
export IFNAME="${NETWORK_INTERFACE}"
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_SOCKET_IFNAME="$IFNAME"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export GLOO_SOCKET_IFNAME="$IFNAME"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve ${WEIGHT_PATH} \
    --host 0.0.0.0 \
    --served-model-name minimax-m3 \
    --trust-remote-code \
    --headless \
    --max-model-len 40960 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --max-num-seqs 8 \
    --data-parallel-size 2 \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank 1 \
    --data-parallel-address $node0_ip \
    --distributed_executor_backend "mp" \
    --gpu-memory-utilization 0.94 \
    --reasoning-parser minimax_m3 \
    --limit-mm-per-prompt '{"image":1,"video":0}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_cpu_binding":true, "ascend_compilation_config":{"fuse_norm_quant":false}, "multistream_overlap_shared_expert": true, "weight_nz_mode": 2}' \
    --port 11223 > ${LOG_PATH} 2>&1 &
```

#### W8A8 Deployment

Run the following command on node 0:

```bash
local_ip="${NODE0_IP}"
node0_ip="${NODE0_IP}"

export HCCL_IF_IP=$local_ip
export IFNAME="${NETWORK_INTERFACE}"
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_SOCKET_IFNAME="$IFNAME"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export GLOO_SOCKET_IFNAME="$IFNAME"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve ${WEIGHT_PATH} \
    --host 0.0.0.0 \
    --served-model-name minimax-m3 \
    --trust-remote-code \
    --max-model-len 131072 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --max-num-seqs 8 \
    --data-parallel-size 2 \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank 0 \
    --data-parallel-address $node0_ip \
    --distributed_executor_backend "mp" \
    --gpu-memory-utilization 0.92 \
    --reasoning-parser minimax_m3 \
    --limit-mm-per-prompt '{"image":1,"video":0}' \
    --speculative-config '{"model":"${EAGLE3_WEIGHT_PATH}", "method":"eagle3", "num_speculative_tokens":3}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_cpu_binding":true, "ascend_compilation_config":{"fuse_norm_quant":false}, "multistream_overlap_shared_expert": false, "weight_nz_mode": 2, "enable_flashcomm1": true}' \
    --port 11223 > ${LOG_PATH} 2>&1 &
```

Run the following command on node 1:

```bash
local_ip="${NODE1_IP}"
node0_ip="${NODE0_IP}"

export HCCL_IF_IP=$local_ip
export IFNAME="${NETWORK_INTERFACE}"
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_SOCKET_IFNAME="$IFNAME"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export GLOO_SOCKET_IFNAME="$IFNAME"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve ${WEIGHT_PATH} \
    --host 0.0.0.0 \
    --served-model-name minimax-m3 \
    --trust-remote-code \
    --headless \
    --max-model-len 131072 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --max-num-seqs 8 \
    --data-parallel-size 2 \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank 1 \
    --data-parallel-address $node0_ip \
    --distributed_executor_backend "mp" \
    --gpu-memory-utilization 0.92 \
    --reasoning-parser minimax_m3 \
    --limit-mm-per-prompt '{"image":1,"video":0}' \
    --speculative-config '{"model":"${EAGLE3_WEIGHT_PATH}", "method":"eagle3", "num_speculative_tokens":3}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_cpu_binding":true, "ascend_compilation_config":{"fuse_norm_quant":false}, "multistream_overlap_shared_expert": false, "weight_nz_mode": 2, "enable_flashcomm1": true}' \
    --port 11223 > ${LOG_PATH} 2>&1 &
```

### 5.3 Prefill-Decode Disaggregation

PD disaggregation separates Prefill and Decode into different service groups. Prefill nodes process large prompt chunks, Decode nodes serve token generation, and a proxy forwards requests between them.

Use Mooncake for deployment. Refer to [Mooncake](../features/pd_disaggregation_mooncake_multi_node.md) for the general PD disaggregation workflow.

Deploy `run_dp_template.sh` and `launch_online_dp.py` on the corresponding nodes, and deploy a proxy script on the prefill node to forward requests.

First, prepare `launch_online_dp.py` on each node:

[launch_online_dp.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/external_online_dp/launch_online_dp.py)

`launch_online_dp.py` passes the visible devices, port, DP size, DP rank, DP address, DP RPC port, and TP size as `$1` through `$7`.

**Common Issues Tip:** For PD disaggregation issues such as KV transfer timeouts or Mooncake connection errors, refer to the [Public FAQs](../../faqs.md). For MiniMax-specific issues, refer to [Chapter 10 FAQ](#10-faq).

=== "A3 series"

    **Hardware**: 2× Atlas 800I A3 (64GB × 16) for `MiniMax-M3-w8a8` with EAGLE3, one for Prefill and one for Decode.

    - 1 Prefill node: 1 Atlas 800I A3 (64G x 16). Runs DP=2, TP=8.
    - 1 Decode node: 1 Atlas 800I A3 (64G x 16). Runs DP=4, TP=4.

    Prefill TP size must be divisible by Decode TP size. The layout above satisfies `8 % 4 == 0`.

    #### Prefill Node

    Create `run_dp_template.sh` on the prefill node. Set `nic_name` and `local_ip` to your own values.

    ```bash
    unset http_proxy https_proxy ftp_proxy

    # Get these values through ifconfig.
    # nic_name is the network interface name corresponding to local_ip.
    nic_name="<your_nic_name>"
    local_ip="<your_ip>"

    export HCCL_BUFFSIZE=1024
    export HCCL_IF_IP=$local_ip
    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_SOCKET_IFNAME=$nic_name
    export ASCEND_RT_VISIBLE_DEVICES=$1

    export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
    export GLOO_SOCKET_IFNAME=$nic_name
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export PYTHONHASHSEED=0

    vllm serve /path/to/weight/MiniMax-M3-w8a8 \
        --host 0.0.0.0 \
        --port $2 \
        --data-parallel-size $3 \
        --data-parallel-rank $4 \
        --data-parallel-address $5 \
        --data-parallel-rpc-port $6 \
        --tensor-parallel-size $7 \
        --enforce-eager \
        --distributed-executor-backend mp \
        --served-model-name minimax-m3 \
        --enable-expert-parallel \
        --seed 1024 \
        --max-model-len 67560 \
        --max-num-batched-tokens 32768 \
        --long-prefill-token-threshold 2048 \
        --trust-remote-code \
        --gpu-memory-utilization 0.95 \
        --reasoning-parser minimax_m3 \
        --dtype bfloat16 \
        --quantization ascend \
        --additional-config '{"enable_cpu_binding":true, "ascend_compilation_config":{"enable_static_kernel":false, "fuse_norm_quant":false}, "multistream_overlap_shared_expert":true, "weight_nz_mode":2, "enable_shared_expert_dp":true, "enable_flashcomm1":true, "enable_reduce_sample":false}' \
        --speculative-config '{"method":"eagle3", "model":"/path/to/weight/MiniMax-M3-EAGLE3", "num_speculative_tokens":3}' \
        --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_port":"30000","engine_id":"0","kv_connector_extra_config":{"use_ascend_direct":true,"prefill":{"dp_size":2,"tp_size":8},"decode":{"dp_size":4,"tp_size":4}}}'
    ```

    Start the Prefill servers:

    ```bash
    python launch_online_dp.py \
        --dp-size 2 --tp-size 8 \
        --dp-size-local 2 --dp-rank-start 0 \
        --dp-address <prefill_ip> --dp-rpc-port 6884 \
        --vllm-start-port 31050
    ```

    This starts two Prefill API servers on ports `31050` and `31051`. `launch_online_dp.py` writes per-rank logs next to the template, such as `out_run_dp_template_rank0.log`. Wait until both ranks print `Application startup complete`.

    Common Issues Tip: If the prefill service is not ready for a long time, check whether the model path is readable on the node, `ASCEND_RT_VISIBLE_DEVICES` covers 8 NPUs per DP rank, and the Mooncake `kv_port` is available.

    #### Decode Node

    Create `run_dp_template.sh` on the decode node.

    ```bash
    unset http_proxy https_proxy ftp_proxy

    # Get these values through ifconfig.
    # nic_name is the network interface name corresponding to local_ip.
    nic_name="<your_nic_name>"
    local_ip="<your_ip>"

    export HCCL_BUFFSIZE=2048
    export HCCL_IF_IP=$local_ip
    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_SOCKET_IFNAME=$nic_name
    export ASCEND_RT_VISIBLE_DEVICES=$1

    export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
    export GLOO_SOCKET_IFNAME=$nic_name
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export PYTHONHASHSEED=0

    vllm serve /path/to/weight/MiniMax-M3-w8a8 \
        --host 0.0.0.0 \
        --port $2 \
        --data-parallel-size $3 \
        --data-parallel-rank $4 \
        --data-parallel-address $5 \
        --data-parallel-rpc-port $6 \
        --tensor-parallel-size $7 \
        --enable-expert-parallel \
        --seed 1024 \
        --served-model-name minimax-m3 \
        --reasoning-parser minimax_m3 \
        --distributed-executor-backend mp \
        --max-model-len 67560 \
        --max-num-batched-tokens 32768 \
        --trust-remote-code \
        --no-enable-prefix-caching \
        --max-num-seqs 64 \
        --gpu-memory-utilization 0.95 \
        --dtype bfloat16 \
        --quantization ascend \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --speculative-config '{"method":"eagle3", "model":"/path/to/weight/MiniMax-M3-EAGLE3", "num_speculative_tokens":3}' \
        --additional-config '{"enable_cpu_binding":true, "ascend_compilation_config":{"enable_static_kernel":false, "fuse_norm_quant":false}, "multistream_overlap_shared_expert":true, "weight_nz_mode":2, "enable_shared_expert_dp":true, "enable_flashcomm1":true, "enable_reduce_sample":false}' \
        --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_consumer","kv_port":"26900","engine_id":"1","kv_connector_extra_config":{"use_ascend_direct":true,"prefill":{"dp_size":2,"tp_size":8},"decode":{"dp_size":4,"tp_size":4}}}'
    ```

    Start the Decode servers:

    ```bash
    python launch_online_dp.py \
        --dp-size 4 --tp-size 4 \
        --dp-size-local 4 --dp-rank-start 0 \
        --dp-address <decode_ip> --dp-rpc-port 5964 \
        --vllm-start-port 31060
    ```

    This starts four Decode API servers on ports `31060` through `31063`.

    Common Issues Tip: If the decode node fails to initialize, check that NPU memory from a previous run has been released, `--tensor-parallel-size` is 4, and `kv_connector_extra_config.decode.dp_size` is 4.

    **A3 layout:** Prefill uses `--data-parallel-size 2` and `--tensor-parallel-size 8`. Decode uses `--data-parallel-size 4` and `--tensor-parallel-size 4`.

=== "Ascend 950DT series"

    **Hardware**: 2× Ascend 950DT (96GB × 8) for `MiniMax-M3-MXFP8` with EAGLE3, one for Prefill and one for Decode.

    - 1 Prefill node: 1 Ascend 950DT (96G x 8). Runs DP=2, TP=4.
    - 1 Decode node: 1 Ascend 950DT (96G x 8). Runs DP=2, TP=4.

    Prefill TP size must be divisible by Decode TP size. The layout above satisfies `4 % 4 == 0`.

    #### Prefill Node

    Create `run_dp_template.sh` on the prefill node. Set `nic_name` and `local_ip` to your own values.

    ```bash
    unset ftp_proxy
    unset https_proxy
    unset http_proxy

    # Get these values through ifconfig.
    # nic_name is the network interface name corresponding to local_ip.
    nic_name="<your_nic_name>"
    local_ip="<your_ip>"

    export HCCL_BUFFSIZE=1024
    export HCCL_IF_IP=$local_ip
    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_SOCKET_IFNAME=$nic_name
    export ASCEND_RT_VISIBLE_DEVICES=$1

    export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH

    export GLOO_SOCKET_IFNAME=$nic_name
    export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
    export PYTHONHASHSEED=0

    vllm serve /path/to/weight/MiniMax-M3-MXFP8 \
      --host 0.0.0.0 \
      --port $2 \
      --data-parallel-size $3 \
      --data-parallel-rank $4 \
      --data-parallel-address $5 \
      --data-parallel-rpc-port $6 \
      --tensor-parallel-size $7 \
      --enforce-eager \
      --served-model-name minimax-m3 \
      --enable-expert-parallel \
      --seed 1024 \
      --max-model-len 67560 \
      --max-num-batched-tokens 32768 \
      --long-prefill-token-threshold 2048 \
      --trust-remote-code \
      --gpu-memory-utilization 0.90 \
      --reasoning-parser minimax_m3 \
      --dtype bfloat16 \
      --quantization mxfp8 \
      --kv-cache-dtype fp8 \
      --kv-cache-dtype-skip-layers 0 1 2 \
      --additional-config '{
        "enable_cpu_binding": true,
        "ascend_compilation_config": {
          "fuse_qknorm_rope": false,
          "fuse_norm_quant": false,
          "enable_static_kernel": false
        },
        "multistream_overlap_shared_expert": true,
        "enable_shared_expert_dp": true,
        "enable_reduce_sample": false
      }' \
      --speculative-config '{
        "method": "eagle3",
        "model": "/path/to/weight/MiniMax-M3-EAGLE3",
        "num_speculative_tokens": 3,
        "kv_cache_dtype": "bfloat16"
      }' \
      --kv-transfer-config '{
        "kv_connector": "MooncakeConnectorV1",
        "kv_role": "kv_producer",
        "kv_port": "30000",
        "engine_id": "0",
        "kv_connector_extra_config": {
          "use_ascend_direct": true,
          "prefill": {"dp_size": 2, "tp_size": 4},
          "decode": {"dp_size": 2, "tp_size": 4}
        }
      }'
    ```

    Start the Prefill servers:

    ```bash
    python launch_online_dp.py \
        --dp-size 2 --tp-size 4 \
        --dp-size-local 2 --dp-rank-start 0 \
        --dp-address <prefill_ip> --dp-rpc-port 6884 \
        --vllm-start-port 31050
    ```

    This starts two Prefill API servers on ports `31050` and `31051`. Wait until both ranks print `Application startup complete`.

    Common Issues Tip: If the prefill service is not ready for a long time, check whether `ASCEND_RT_VISIBLE_DEVICES` covers 4 NPUs per DP rank and the Mooncake `kv_port` is available.

    #### Decode Node

    Create `run_dp_template.sh` on the decode node.

    ```bash
    unset ftp_proxy
    unset https_proxy
    unset http_proxy

    # Get these values through ifconfig.
    # nic_name is the network interface name corresponding to local_ip.
    nic_name="<your_nic_name>"
    local_ip="<your_ip>"

    export HCCL_BUFFSIZE=2048
    export HCCL_IF_IP=$local_ip
    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_SOCKET_IFNAME=$nic_name
    export ASCEND_RT_VISIBLE_DEVICES=$1

    export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH

    export GLOO_SOCKET_IFNAME=$nic_name
    export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
    export PYTHONHASHSEED=0

    vllm serve /path/to/weight/MiniMax-M3-MXFP8 \
      --host 0.0.0.0 \
      --port $2 \
      --data-parallel-size $3 \
      --data-parallel-rank $4 \
      --data-parallel-address $5 \
      --data-parallel-rpc-port $6 \
      --tensor-parallel-size $7 \
      --enable-expert-parallel \
      --seed 1024 \
      --served-model-name minimax-m3 \
      --reasoning-parser minimax_m3 \
      --distributed-executor-backend mp \
      --max-model-len 67560 \
      --max-num-batched-tokens 32768 \
      --trust-remote-code \
      --no-enable-prefix-caching \
      --max-num-seqs 64 \
      --gpu-memory-utilization 0.90 \
      --dtype bfloat16 \
      --quantization mxfp8 \
      --kv-cache-dtype fp8 \
      --kv-cache-dtype-skip-layers 0 1 2 \
      --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
      --speculative-config '{"method":"eagle3","model":"/path/to/weight/MiniMax-M3-EAGLE3","num_speculative_tokens":3,"kv_cache_dtype": "bfloat16"}' \
      --additional-config '{
        "enable_cpu_binding": true,
        "ascend_compilation_config": {
          "enable_static_kernel": false,
          "fuse_norm_quant": false
        },
        "multistream_overlap_shared_expert": true,
        "enable_shared_expert_dp": true,
        "enable_flashcomm1": false,
        "enable_reduce_sample": false
      }' \
      --kv-transfer-config '{
        "kv_connector": "MooncakeConnectorV1",
        "kv_role": "kv_consumer",
        "kv_port": "26900",
        "engine_id": "1",
        "kv_connector_extra_config": {
          "use_ascend_direct": true,
          "prefill": {"dp_size": 2, "tp_size": 4},
          "decode": {"dp_size": 2, "tp_size": 4}
        }
      }'
    ```

    Start the Decode servers:

    ```bash
    python launch_online_dp.py \
        --dp-size 2 --tp-size 4 \
        --dp-size-local 2 --dp-rank-start 0 \
        --dp-address <decode_ip> --dp-rpc-port 5964 \
        --vllm-start-port 31060
    ```

    This starts two Decode API servers on ports `31060` and `31061`.

    Common Issues Tip: If the decode node fails to initialize, check that `--tensor-parallel-size` is 4 and `kv_connector_extra_config.decode.dp_size` is 2.

    **950DT layout:** Prefill and Decode both use `--data-parallel-size 2` and `--tensor-parallel-size 4`.

**Key parameters for PD disaggregation:**

- `launch_online_dp.py` starts one `vllm serve` process per local DP rank and sets `ASCEND_RT_VISIBLE_DEVICES` to a TP-sized slice of the node.
- `--data-parallel-address` and `--data-parallel-rpc-port` define the DP control plane on each node.
- `--distributed-executor-backend mp` uses multiprocessing for the local TP workers.
- `--kv-transfer-config` sets the Mooncake connector. `kv_role` is `kv_producer` on prefill and `kv_consumer` on decode.
- `kv_connector_extra_config.prefill.dp_size/tp_size` and `decode.dp_size/tp_size` must match the actual global DP and TP layout on both nodes.
- Prefill TP size must be divisible by Decode TP size.
- `--no-enable-prefix-caching` disables prefix caching. For PD disaggregation, the D-node prefix-cache known issue is tracked in [#7944](https://github.com/vllm-project/vllm-ascend/issues/7944).
- `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` is recommended on the decode node to reduce decode dispatch overhead.
- `--enforce-eager` is used on the prefill node.

### 5.4 Request Forwarding

Run a proxy server on a node that can reach all Prefill and Decode API ports. You can get the proxy program from the repository: [load_balance_proxy_server_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py).

Wait until every Prefill and Decode rank prints `Application startup complete` before starting the proxy.

=== "A3 series"

    For A3 PD disaggregation (1P1D), the proxy forwards requests to 2 Prefill ranks and 4 Decode ranks.

    ```bash
    unset ftp_proxy
    unset https_proxy
    unset http_proxy

    python load_balance_proxy_server_example.py \
        --port 8009 \
        --host 0.0.0.0 \
        --prefiller-hosts \
           <prefill_ip> <prefill_ip> \
        --prefiller-ports \
           31050 31051 \
        --decoder-hosts \
           <decode_ip> <decode_ip> <decode_ip> <decode_ip> \
        --decoder-ports \
           31060 31061 31062 31063
    ```

=== "Ascend 950DT series"

    For Ascend 950DT PD disaggregation (1P1D), the proxy forwards requests to 2 Prefill ranks and 2 Decode ranks.

    ```bash
    unset ftp_proxy
    unset https_proxy
    unset http_proxy

    python load_balance_proxy_server_example.py \
        --port 8009 \
        --host 0.0.0.0 \
        --prefiller-hosts \
           <prefill_ip> <prefill_ip> \
        --prefiller-ports \
           31050 31051 \
        --decoder-hosts \
           <decode_ip> <decode_ip> \
        --decoder-ports \
           31060 31061
    ```

The service is then accessible at `http://<proxy_ip>:8009`. For PD disaggregation, use this proxy endpoint in Section 6.

Common Issues Tip: If requests reach the proxy but no output is returned, check that the proxy host list includes every healthy Prefill and Decode port, and that both nodes still have free NPU memory after the previous run.

### 5.5 Multimodal and ViT DP (Optional)

MiniMax-M3 supports image and video inputs on Ascend. The deployment examples above keep `--limit-mm-per-prompt '{"image":1,"video":0}'` as the default multimodal capacity assumption because the other serving parameters are tuned for the single-image path.

MiniMax-M3 image and video inputs share the same Vision Tower. If a service only needs one modality, explicitly set the unused modality to `0`; for example, use `{"image":1,"video":0}` for image-only serving and `{"image":0,"video":1}` for video-only serving. As long as either image or video remains enabled, the shared Vision Tower is retained. Setting an unused modality to `0` is clearer than omitting it, because omitted modalities may still participate in multimodal capacity and profiling planning.

For the ViT / multimodal encoder part, data parallel execution is supported and can be enabled with:

```bash
--mm-encoder-tp-mode data
```

This option is not enabled in the default deployment examples because it can increase per-card memory usage. When enabling ViT DP, re-evaluate memory-related parameters such as `--max-model-len`, `--max-num-seqs`, and `--gpu-memory-utilization` for the target workload.

For video or mixed image-video requests, adjust the multimodal limit according to the actual request shape instead of changing the default template blindly:

```bash
# one video
--limit-mm-per-prompt '{"image":0,"video":1}'

# one image and one video
--limit-mm-per-prompt '{"image":1,"video":1}'
```

When using local media paths in requests, such as `file:///path/to/video.mp4`, add an explicit allowlist path:

```bash
--allowed-local-media-path /
```

If the number of sampled video frames is not specified, vLLM uses its default video sampling policy, which samples 32 frames by default. For quick functional smoke tests, a smaller frame count such as 8 or 16 can be set in the request or evaluation config. For benchmark runs, follow the dataset protocol.

FLASHCOMM1 and language-model-only mode should not be enabled at the same time for MiniMax-M3 serving. FLASHCOMM1 is enabled through `additional_config.enable_flashcomm1`, while language-model-only mode is enabled with `--language-model-only`.

```bash
# Enable FLASHCOMM1.
--additional-config '{"enable_flashcomm1": true}'

# Enable language-model-only mode.
--language-model-only
```

`VLLM_ASCEND_ENABLE_FLASHCOMM1=1` is kept for compatibility, but `additional_config.enable_flashcomm1` is preferred.

## 6 Thinking and Parser Configuration

### 6.1 Thinking Mode

MiniMax-M3 supports three thinking modes, controlled via `thinking_mode` in `chat_template_kwargs`:


| Mode       | Behavior                                                                  | Use Case                  |
| ---------- | ------------------------------------------------------------------------- | ------------------------- |
| `enabled`  | The model thinks before every response, including after tool results      | Complex reasoning, agents |
| `disabled` | No thinking; the model answers directly                                   | Latency-sensitive turns   |
| `adaptive` | The model decides whether to think based on the task (default when unset) | General use               |


#### 6.1.1 Request Examples

**With thinking disabled (curl):**

```bash
curl http://{ip}:{port}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minimax-m3",
    "messages": [{"role": "user", "content": "who are you?"}],
    "max_tokens": 100,
    "stream": false,
    "top_p": 0.95,
    "top_k": 40,
    "temperature": 1.0,
    "chat_template_kwargs": {"thinking_mode": "disabled"}
  }'
```

Change `"thinking_mode"` to `"enabled"` or `"adaptive"` as needed. The deprecated `enable_thinking` parameter (equivalent to `thinking_mode: "enabled"`) is also supported.

**With thinking enabled (Python SDK):**

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

response = client.chat.completions.create(
    model="minimax-m3",
    messages=[{"role": "user", "content": "Prove there are infinitely many primes."}],
    extra_body={"chat_template_kwargs": {"thinking_mode": "enabled"}},
)
msg = response.choices[0].message
print(getattr(msg, "reasoning", None))  # the <mm:think> block
print(msg.content)                       # the final answer
```

### 6.2 Reasoning Parser

The MiniMax-M3 reasoning parser (`--reasoning-parser minimax_m3`) extracts the thinking block `<mm:think>...</mm:think>` from model output and exposes it as the `reasoning` field. The remaining text is returned as `content`.

#### 6.2.1 Server Configuration

The `--reasoning-parser minimax_m3` flag enables the MiniMax-M3 reasoning parser, which splits model output into reasoning and content using `<mm:think>...</mm:think>` delimiters:

```bash
vllm serve ${WEIGHT_PATH} \
  --reasoning-parser minimax_m3 \
  ...
```

#### 6.2.2 Output Format

MiniMax-M3 uses explicit thinking delimiters:

```text
<mm:think>reasoning process...</mm:think>final answer
```

#### 6.2.3 Parser Behavior

- `**thinking_mode="enabled"**`: The chat template pre-fills `<mm:think>` in the prompt. Generated text starts inside the reasoning block and transitions to content after `</mm:think>`.
- `**thinking_mode="disabled"` or default**: Model output is treated as plain content. If `<mm:think>` appears, the parser splits on the delimiters.
- **Streaming**: Reasoning and content are streamed incrementally via `DeltaMessage.reasoning` and `DeltaMessage.content` token-by-token.
- **Token counting**: Reasoning tokens inside `<mm:think>` blocks are correctly counted.

### 6.3 Tool Call Parser

MiniMax-M3 uses a namespace-delimited XML format for tool calls. Enable it with `--tool-parser minimax_m3`.

#### 6.3.1 Server Configuration

When both `--reasoning-parser minimax_m3` and `--tool-call-parser minimax_m3` are specified, the parsers work together automatically to handle responses that contain both reasoning blocks and tool calls:

```bash
vllm serve ${WEIGHT_PATH} \
  --reasoning-parser minimax_m3 \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m3 \
  ...
```

#### 6.3.2 Tool Call Format

Each structural tag is preceded by the `]<]minimax[>[` namespace marker:

```xml
]<]minimax[>[<tool_call>
]<]minimax[>[<invoke name="create_order">
]<]minimax[>[<user_id>42]<]minimax[>[</user_id>
]<]minimax[>[<shipping>
]<]minimax[>[<city>Singapore]<]minimax[>[</city>
]<]minimax[>[<zip>018956]<]minimax[>[</zip>
]<]minimax[>[</shipping>
]<]minimax[>[</invoke>
]<]minimax[>[</tool_call>
```

#### 6.3.3 Key Features

- **Recursive parameter parsing**: Supports nested objects and arrays (e.g., `shipping` containing `city`/`zip`).
- **Schema-aware type coercion**: String parameter values are automatically converted to the correct types (integer, boolean, object, array) based on the function's JSON Schema definition.
- **Multiple invocations**: A single `<tool_call>` block can contain multiple `<invoke>` blocks.
- **Streaming**: Tool name and argument fragments are streamed incrementally as the `<invoke>` block is received.

#### 6.3.4 Request Example (curl)

```bash
curl http://{ip}:{port}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minimax-m3",
    "messages": [{"role": "user", "content": "What's the weather like in Shanghai?"}],
    "max_tokens": 300,
    "stream": false,
    "tool_choice": "auto",
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City or country name"
                        }
                    },
                    "required": ["location"],
                    "additionalProperties": false
                }
            }
        }
    ],
    "chat_template_kwargs": {"thinking_mode": "disabled"}
  }'
```

## 7 Functional Verification

### 7.1 Text

  Expected result: the answer is C.

### 7.2 Single Image

  Start the service with image input enabled, for example `--limit-mm-per-prompt '{"image":1,"video":0}'`. Replace `${IMAGE_PATH}` with a local image path on the client side.

### 7.3 Single Video

  Start the service with video input enabled, for example `--limit-mm-per-prompt '{"image":0,"video":1}'`. If the request uses `file://` local video paths, also add `--allowed-local-media-path /` or a narrower allowed directory. If `media_io_kwargs.video.num_frames` is not specified, vLLM samples 32 frames by default.

### 7.4 Mixed Image and Video Request

  Start the service with both image and video input enabled. For the following request, use `--limit-mm-per-prompt '{"image":1,"video":1}'`. If the request uses `file://` local video paths, also add `--allowed-local-media-path /` or a narrower allowed directory.

## 8 Accuracy Evaluation

### 8.1 Using AISBench

For detailed instructions, refer to [Using AISBench for accuracy evaluation](../../developer_guide/evaluation/using_ais_bench.md).

### 8.2 Text Evaluation


| Dataset      | Hardware | Score        | max-model-len | max-num-seqs | max_out_len | batch_size | generation_kwargs           |
| ------------ | -------- | ------------ | ------------- | ------------ | ----------- | ---------- | --------------------------- |
| GSM8K        | GPU      | 96.72        | 65536         | 16           | 49152       | 16         | temperature=1.0, top_p=0.95 |
| GSM8K        | NPU      | 96.36        | 10240         | 16           | 9500        | 20         | temperature=1.0, top_p=0.95 |
| AIME2025     | GPU      | 95@repeat4   | -             | -            | -           | -          | -                           |
| AIME2025     | NPU      | 93.3@repeat2 | 131072        | 32           | 65536       | 8          | temperature=1.0, top_p=0.95 |
| GPQA-Diamond | GPU      | 92.42        | 81920         | 64           | 75776       | 8          | temperature=0.6, top_p=0.95 |
| GPQA-Diamond | NPU      | 92.42        | 131072        | 32           | 65536       | 8          | temperature=0.6, top_p=0.95 |


### 8.3 Multimodal Evaluation

MiniMax-M3 multimodal accuracy is evaluated with AISBench. The ViT DP path is optional and can be enabled by adding `--mm-encoder-tp-mode data` to the serving command, but it is not required for all multimodal accuracy runs. For video evaluation, if no frame count is specified in the request or evaluation config, vLLM samples 32 frames by default.

The Video-MME results below are measured on chunk1 and chunk2, not the full dataset.

For Video-MME evaluation, run the vLLM OpenAI-compatible service with video input enabled and use AISBench to send the Video-MME requests. The official AISBench guide may not list Video-MME as a built-in example, so the key MiniMax-M3 settings used here are:

- serve with `--limit-mm-per-prompt '{"image":0,"video":1}'`;
- do not set `media_io_kwargs.video.num_frames`, so vLLM uses the default 32 sampled frames;
- use `max-model-len=90112` and `max_out_len=8192`;
- evaluate Video-MME chunk1 and chunk2, not the full dataset.

The AISBench command used for the Video-MME chunk1+chunk2 evaluation is:

```bash
ais_bench \
  --models vllm_api_general_chat \
  --datasets videomme_subset_1_2.py \
  --mode all \
  --dump-eval-details \
  --merge-ds
```

`videomme_subset_1_2.py` is a local AISBench dataset config derived from the original Video-MME config, such as `videomme_gen.py`. It points `path` to the parquet file filtered from the full Video-MME metadata by the locally available chunk1/chunk2 videos, and points `video_path` to the extracted chunk1/chunk2 `.mp4` directory. This keeps the evaluation lightweight while preserving the standard Video-MME request and scoring flow.


| Dataset                 | Modality | Tool     | Hardware | ViT DP   | max-model-len | max_out_len | Input Config                                                       | generation_kwargs           | Score |
| ----------------------- | -------- | -------- | -------- | -------- | ------------- | ----------- | ------------------------------------------------------------------ | --------------------------- | ----- |
| TextVQA                 | Image    | AISBench | GPU      | disabled | 65536         | 512         | `--limit-mm-per-prompt '{"image":1,"video":0}'`                    | temperature=1.0, top_p=0.95 | 70.82 |
| TextVQA                 | Image    | AISBench | NPU      | disabled | 65536         | 512         | `--limit-mm-per-prompt '{"image":1,"video":0}'`                    | temperature=1.0, top_p=0.95 | 72.75 |
| Video-MME chunk1+chunk2 | Video    | AISBench | GPU      | -        | 90112         | 8192        | `--limit-mm-per-prompt '{"image":0,"video":1}'`, default 32 frames | temperature=1.0, top_p=0.95 | 73.41 |
| Video-MME chunk1+chunk2 | Video    | AISBench | NPU      | -        | 90112         | 8192        | `--limit-mm-per-prompt '{"image":0,"video":1}'`, default 32 frames | temperature=1.0, top_p=0.95 | 74.21 |


## 9 Performance Tuning

> **Note**: The following configurations are validated in specific test environments and are for reference only. The optimal configuration depends on factors such as maximum input/output length, prefix cache hit rate, precision requirements, and deployment machine ratios. It is recommended to refer to Section 9.2 for tuning based on actual conditions.

### 9.1 Recommended Configurations

The recommended configurations are the same as those specified in Chapter 5, “Online Service Deployment.”

### 9.2 Tuning Guidelines

#### 9.2.1 General Tuning Reference

Please refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for general tuning methods.

Please refer to the [Feature Matrix](../../user_guide/support_matrix/feature_matrix.md) for detailed feature descriptions.

## 10 FAQ

**Q: How can I reinstall vLLM Ascend?**

Use the following command to reinstall vLLM Ascend and build it with the dependencies from the current Python environment:

```bash
pip install -v --no-build-isolation -e . -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
```

**Q: What should I do if a video request is slow or times out when `media_io_kwargs.video.num_frames` is not set?**

By default, vLLM samples 32 frames when reading a video. MiniMax-M3 produces many visual tokens per frame, so a 32-frame video significantly increases prefill computation. If the request is slow or times out, explicitly set `media_io_kwargs.video.num_frames` to a smaller value, such as 8 or 16 frames:

```json
{
  "media_io_kwargs": {
    "video": {
      "num_frames": 8
    }
  }
}
```

