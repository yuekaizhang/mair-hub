stage=${1:-0}
stop_stage=${2:-99}

echo "Start stage: $stage, Stop stage: $stop_stage"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:/workspace_yuekai/tts/ZipVoice

model_name=zipvoice_distill
MODEL_DIR=models
VOCODER_ONNX_PATH=$MODEL_DIR/vocoder/vocos_vocoder.onnx
VOCODER_TRT_ENGINE_PATH=$MODEL_DIR/vocoder/vocos_vocoder.plan
mkdir -p $MODEL_DIR/vocoder
MODEL_REPO=./model_repo_zipvoice

if [ "$stage" -le 1 ] && [ "$stop_stage" -ge 1 ]; then
    echo "Stage 1: Download prompt audio"
    pip install -r requirements.txt
    huggingface-cli download k2-fsa/ZipVoice --local-dir $MODEL_DIR
fi

if [ "$stage" -le 2 ] && [ "$stop_stage" -ge 2 ]; then
    echo "Stage 2: Export Zipvoice TensorRT model"
    python3 -m zipvoice.bin.tensorrt_export \
        --model-name $model_name \
        --model-dir $MODEL_DIR/$model_name \
        --checkpoint-name model.pt \
        --trt-engine-file-name fm_decoder.fp16.plan \
        --tensorrt-model-dir $MODEL_DIR/${model_name}_trt || exit 1
fi



if [ "$stage" -le 3 ] && [ "$stop_stage" -ge 3 ]; then
    echo "Building triton server"
    rm -r $MODEL_REPO
    cp -r ./model_repo $MODEL_REPO
    python3 scripts/fill_template.py -i $MODEL_REPO/zipvoice/config.pbtxt model_dir:$MODEL_DIR/$model_name,model_name:$model_name,trt_engine_path:$MODEL_DIR/${model_name}_trt/fm_decoder.fp16.plan
fi

if [ "$stage" -le 4 ] && [ "$stop_stage" -ge 4 ]; then
    echo "Starting triton server"
    tritonserver --model-repository=$MODEL_REPO
fi

if [ "$stage" -le 5 ] && [ "$stop_stage" -ge 5 ]; then
    echo "Testing triton server"
    num_task=4
    split_name=wenetspeech4tts
    log_dir=./log_pytriton_vocoder_${model_name}_concurrent_${num_task}_${split_name}
    python3 client_grpc.py --num-tasks $num_task --huggingface-dataset yuekai/seed_tts --split-name $split_name --log-dir $log_dir
fi

if [ "$stage" -le 6 ] && [ "$stop_stage" -ge 6 ]; then
    echo "Testing http client"
    wget -nc https://raw.githubusercontent.com/SparkAudio/Spark-TTS/main/example/prompt_audio.wav -O prompt.wav
    python3 client_http.py --reference-audio prompt.wav \
        --reference-text "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。" \
        --target-text "身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。" \
        --output-audio "./test.wav"
fi


if [ "$stage" -le 7 ] && [ "$stop_stage" -ge 7 ]; then
    python3 infer_zipvoice_trt.py \
        --model-name zipvoice_distill \
        --huggingface-dataset-name yuekai/seed_tts_cosy2 \
        --huggingface-dataset-split wenetspeech4tts \
        --trt-engine-path models/zipvoice_distill_trt/fm_decoder.fp16.plan \
        --batch-size 4 \
        --enable-warmup True \
        --num-step 4 \
        --res-dir results
fi

if [ "$stage" -le 8 ] && [ "$stop_stage" -ge 8 ]; then
    export CUDA_VISIBLE_DEVICES=1
    python3 pytriton_server.py  \
        --model_dir models/zipvoice_distill \
        --model_name zipvoice_distill \
        --trt_engine_path models/zipvoice_distill_trt/fm_decoder.fp16.plan \
        --reference_audio_sample_rate 16000 \
        --port 8000 \
        --max_batch_size 4 \
        --verbose
fi