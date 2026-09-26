#!/bin/bash
# RunPod(PyTorch 템플릿, L40S) 초기화: vLLM 0.26.0 설치 + 모델 다운로드 + dev 실측 대기열.
# 사용: HF_TOKEN=... bash pod_bootstrap.sh     (코드·데이터는 /workspace/nara 에 먼저 올려 둔다)
# 로그: /workspace/logs/{setup,download,run_dev}.log
set -u
mkdir -p /workspace/logs /workspace/models
cd /workspace

cat > /workspace/setup.sh <<'SH'
set -x
pip install -q -U uv
uv venv -p 3.12 --clear /workspace/venv
. /workspace/venv/bin/activate
export UV_CONCURRENT_DOWNLOADS=8 UV_HTTP_TIMEOUT=300
uv pip install "vllm==0.26.0" --torch-backend=cu130 huggingface_hub hf_transfer pandas
python -c "import vllm,torch,transformers;print('VERSIONS',vllm.__version__,torch.__version__,transformers.__version__)"
SH

cat > /workspace/download.sh <<'SH'
set -x
pip install -q -U huggingface_hub hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download google/gemma-4-26B-A4B-it --revision 4d7ae4984b7db7de8f8457170b3f1a419ee76d52 --local-dir /workspace/models/gemma-4-26B-A4B-it
echo DOWNLOAD_DONE
SH

cat > /workspace/run_dev.sh <<'SH'
cd /workspace/nara
until grep -q VERSIONS /workspace/logs/setup.log && grep -q DOWNLOAD_DONE /workspace/logs/download.log; do
  pgrep -f setup.sh >/dev/null || grep -q VERSIONS /workspace/logs/setup.log || { echo SETUP_FAILED; exit 1; }
  sleep 20
done
. /workspace/venv/bin/activate
export PPS_MODEL_DIR=/workspace/models/gemma-4-26B-A4B-it PYTHONIOENCODING=utf-8
run() { tag=$1; shift; echo "=== RUN $tag $(date)"; python tools/gpu_dev_run.py --tag "$tag" "$@" > /workspace/logs/dev_$tag.log 2>&1 || echo "RUN_FAILED $tag"; tail -n 1 /workspace/logs/dev_$tag.log; }
run s003  --script submissions/003_20260926/script.py
run s003g --script submit/script.py --set USE_ITEM_GUIDE=true
run s003grp --script submit/script.py --set GROUP_MODE=true
run s003fct --script submit/script.py --set GROUP_MODE=true --set FACTS_MODE=true
echo ALL_DONE
SH

(service ssh start || /usr/sbin/sshd) >/dev/null 2>&1
setsid nohup bash /workspace/setup.sh > /workspace/logs/setup.log 2>&1 < /dev/null &
HF_TOKEN="${HF_TOKEN:?HF_TOKEN 필요}" setsid nohup bash /workspace/download.sh > /workspace/logs/download.log 2>&1 < /dev/null &
setsid nohup bash /workspace/run_dev.sh > /workspace/logs/run_dev.log 2>&1 < /dev/null &
echo "bootstrap started"
