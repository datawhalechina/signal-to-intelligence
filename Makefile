.PHONY: help install install-ml install-llm install-rag install-hf install-vision install-all \
        test test-cov lint lint-ch04 lint-project-signal-detection lint-project-vision-detection format \
        run-jupyter run-exp-ch01 run-exp-ch02 run-exp-ch03 run-exp-ch04 \
        run-exp-ch05 run-exp-ch05-api run-exp-ch06 \
        run-exp-ch07 run-exp-ch07-heavy run-exp-ch08 run-project-signal-detection \
        run-project-signal-detection-cnn run-project-vision-detection run-all-exp \
        docker-up docker-up-gpu docker-down docker-logs \
        clean clean-cache

help:
	@echo ""
	@echo "Signals to Intelligence"
	@echo "======================="
	@echo ""
	@echo "环境安装："
	@echo "  make install          安装核心依赖（第1-5章离线实验，CPU）"
	@echo "  make install-ml       安装深度学习依赖（第3-4章，需要 torch）"
	@echo "  make install-llm      安装 LLM API 依赖（第5-6章，需要 API Key）"
	@echo "  make install-rag      安装 RAG 依赖（第6章，langchain/faiss）"
	@echo "  make install-hf       安装 Hugging Face 生态（本地开源模型）"
	@echo "  make install-vision   安装 YOLO/DETR 视觉检测项目依赖"
	@echo "  make install-all      安装全部依赖"
	@echo ""
	@echo "代码实验（离线，无需 API Key / GPU）："
	@echo "  make run-exp-ch01     第1章：FFT、位置编码（主力 2 个）"
	@echo "  make run-exp-ch02     第2章：LMS vs Adam、MMSE（主力 2 个）"
	@echo "  make run-exp-ch03     第3章：MLP、CNN、RNN（主力 3 个）"
	@echo "  make run-exp-ch04     第4章：自注意力（主力 1 个）"
	@echo "  make run-exp-ch05     第5章：全部离线实验（8 个）"
	@echo "  make run-exp-ch06     第6章：Prompt、微调、RAG、Agent 与系统选型（5 个）"
	@echo "  make run-exp-ch07     第7章：ViT、CLIP、高分辨率与评测图（8 个）"
	@echo "  make run-exp-ch08     第8章：LLM 工程实践（标准库）"
	@echo "  make run-project-signal-detection  项目：信号检测方法对照与失败案例"
	@echo "  make run-project-vision-detection  项目：YOLO/DETR 对比（需 ANNOTATIONS、IMAGES_DIR）"
	@echo "  make run-all-exp      运行所有离线实验"
	@echo ""
	@echo "代码实验（需要 API Key / GPU / 额外依赖）："
	@echo "  make run-exp-ch05-api       第5章：LLM API 演示（需要 Ollama 或 API Key）"
	@echo "  make run-exp-ch07-heavy     第7章：真实 CLIP 对齐演示（需 GPU + openai-clip）"
	@echo "  make run-project-signal-detection-cnn  项目扩展：轻量 1D CNN（需 torch）"
	@echo ""
	@echo "测试："
	@echo "  make test             运行所有测试"
	@echo "  make test-cov         运行测试并生成覆盖率报告"
	@echo "  make lint-ch04        第4章试点代码与测试的静态检查"
	@echo "  make lint-project-signal-detection  信号检测项目的静态检查"
	@echo "  make lint-project-vision-detection  视觉检测项目评估层的静态检查"
	@echo ""
	@echo "Docker（配置文件在 deploy/ 目录）："
	@echo "  make docker-up        启动 CPU 版 Jupyter（第1-5章）"
	@echo "  make docker-up-gpu    启动 GPU 版 Jupyter（第5-6章开源模型）"
	@echo "  make docker-down      停止容器"
	@echo "  make docker-logs      查看容器日志"
	@echo ""
	@echo "清理："
	@echo "  make clean            删除构建产物"
	@echo "  make clean-cache      删除缓存文件"
	@echo ""

# ── 环境安装 ──────────────────────────────────────────────

install:
	pip install -e .

install-ml:
	pip install -e ".[ml]"

install-llm:
	pip install -e ".[llm]"

install-rag:
	pip install -e ".[rag]"

install-hf:
	pip install -e ".[hf]"

install-vision:
	pip install -e ".[vision]"

install-all:
	pip install -e ".[all]"

# ── 代码实验 ──────────────────────────────────────────────

run-exp-ch01:
	python code/ch01_dsp/fft_spectrum.py
	python code/ch01_dsp/positional_encoding.py

run-exp-ch02:
	python code/ch02_optimization/lms_vs_adam.py
	python code/ch02_optimization/mmse_vs_nn.py

run-exp-ch03:
	python code/ch03_deep_learning_fast/polynomial_vs_mlp.py
	python code/ch03_deep_learning_fast/mnist_cnn.py
	python code/ch03_deep_learning_fast/rnn_structure.py

run-exp-ch04:
	python code/ch04_transformer/self_attention.py

# 第5章：全部离线实验（不需要 API Key，不需要 GPU）
run-exp-ch05:
	python code/ch05_llm_basics/bpe_tokenization.py
	python code/ch05_llm_basics/scaling_laws.py
	python code/ch05_llm_basics/autoregressive_generation.py
	python code/ch05_llm_basics/training_data_composition.py
	python code/ch05_llm_basics/model_families_evolution.py
	python code/ch05_llm_basics/lora_visualization.py
	python code/ch05_llm_basics/rlhf_pipeline.py
	python code/ch05_llm_basics/benchmark_comparison.py

# 第5章：API 演示（需要本地 Ollama 或任一受支持 API Key）
run-exp-ch05-api:
	python code/ch05_llm_basics/llm_api_demo.py

run-exp-ch06:
	python code/ch06_llm_applications/prompt_demo.py
	python code/ch06_llm_applications/finetuning_demo.py
	python code/ch06_llm_applications/rag_demo.py
	python code/ch06_llm_applications/agent_demo.py
	python code/ch06_llm_applications/system_design_demo.py

run-exp-ch07:
	python code/ch07_multimodal_llm/vit_patches.py
	python code/ch07_multimodal_llm/clip_similarity.py
	python code/ch07_multimodal_llm/high_resolution_processing.py
	python code/ch07_multimodal_llm/qwen_vl_analysis.py
	python code/ch07_multimodal_llm/architecture_diagrams.py
	python code/ch07_multimodal_llm/explainer_diagrams.py
	python code/ch07_multimodal_llm/multimodal_applications.py
	python code/ch07_multimodal_llm/case_studies.py

# 第7章：重度实验（需要 GPU + pip install -e ".[hf]" + 下载模型）
# clip_alignment_demo 还需要 openai-clip：pip install openai-clip
run-exp-ch07-heavy:
	python code/ch07_multimodal_llm/clip_alignment_demo.py

# 第8章
run-exp-ch08:
	python code/ch08_llm_engineering/llm_engineering_demo.py

run-project-signal-detection:
	python projects/signal_detection/main.py

run-project-signal-detection-cnn:
	python projects/signal_detection/cnn_main.py

run-project-vision-detection:
	python -m projects.vision_detection_comparison.main \
		--annotations "$(ANNOTATIONS)" --images-dir "$(IMAGES_DIR)" \
		--device "$(or $(DEVICE),cpu)" --limit "$(or $(LIMIT),50)"

# 运行所有离线实验（不需要 API Key，不需要 GPU）
run-all-exp:
	@echo "运行所有离线实验..."
	$(MAKE) run-exp-ch01
	$(MAKE) run-exp-ch02
	$(MAKE) run-exp-ch03
	$(MAKE) run-exp-ch04
	$(MAKE) run-exp-ch05
	$(MAKE) run-exp-ch06
	$(MAKE) run-exp-ch07
	$(MAKE) run-exp-ch08
	@echo "完成。图表已保存到 assets/ 目录。"

run-jupyter:
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

# ── 测试 ──────────────────────────────────────────────────

test:
	env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q

test-cov:
	env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p pytest_cov tests/ -q --cov=code --cov-report=html --cov-report=term-missing

lint:
	flake8 code tests --max-line-length=100
	mypy code --ignore-missing-imports

# 第4章采用核心计算与演示入口分层，作为渐进式质量门禁试点。
lint-ch04:
	flake8 code/ch04_transformer tests/test_ch04_transformer.py --max-line-length=100

lint-project-signal-detection:
	flake8 projects/signal_detection tests/test_signal_detection_project.py \
		tests/test_signal_detection_cnn.py --max-line-length=100

lint-project-vision-detection:
	flake8 projects/vision_detection_comparison tests/test_vision_detection_project.py --max-line-length=100

format:
	black code tests
	isort code tests

# ── Docker（配置文件在 deploy/）────────────────────────────

docker-up:
	docker compose -f deploy/docker-compose.yml up -d
	@echo "Jupyter 已启动：http://localhost:8888"

docker-up-gpu:
	docker compose -f deploy/docker-compose.gpu.yml up -d
	@echo "GPU 版 Jupyter 已启动：http://localhost:8888"

docker-down:
	docker compose -f deploy/docker-compose.yml down
	docker compose -f deploy/docker-compose.gpu.yml down 2>/dev/null || true

docker-logs:
	docker compose -f deploy/docker-compose.yml logs -f jupyter

# ── 清理 ──────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/

clean-cache:
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
