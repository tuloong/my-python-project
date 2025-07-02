# Stock Prediction Project Makefile

.PHONY: help install test lint format clean data train predict setup-env run-tests

# 默认目标
help:
	@echo "可用命令:"
	@echo "  install     - 安装项目依赖"
	@echo "  setup-env   - 设置conda环境"
	@echo "  test        - 运行所有测试"
	@echo "  lint        - 代码风格检查"
	@echo "  format      - 格式化代码"
	@echo "  clean       - 清理临时文件"
	@echo "  data        - 准备数据"
	@echo "  train       - 训练模型"
	@echo "  predict     - 生成预测"
	@echo "  run-tests   - 运行单元测试"

# 安装依赖
install:
	pip install -r requirements.txt

# 设置conda环境
setup-env:
	conda env create -f environment.yml
	conda activate stock-prediction

# 运行测试
test:
	python -m pytest tests/ -v --cov=src --cov-report=html

# 代码风格检查
lint:
	flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503
	mypy src/ --ignore-missing-imports

# 格式化代码
format:
	black src/ tests/ --line-length=100
	isort src/ tests/ --profile black

# 清理临时文件
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage

# 准备数据
data:
	python prepare_data.py

# 训练模型
train:
	python train_model.py

# 生成预测
predict:
	python predict.py

# 运行单元测试
run-tests:
	python -m unittest discover tests/ -v

# 创建目录结构
setup-dirs:
	mkdir -p data/raw data/processed data/external
	mkdir -p outputs/models outputs/predictions
	mkdir -p logs
	mkdir -p cache

# 下载示例数据
download-sample:
	@echo "下载示例数据..."
	python -c "
import yfinance as yf
import pandas as pd
import os

# 获取沪深300成分股（简化版本）
tickers = ['000001.SZ', '000002.SZ', '600000.SS', '600036.SS', '600519.SS']
start_date = '2020-01-01'
end_date = '2023-12-31'

os.makedirs('data/raw', exist_ok=True)

for ticker in tickers:
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        data.to_csv(f'data/raw/{ticker}.csv')
        print(f'已下载 {ticker} 数据')
    except Exception as e:
        print(f'下载 {ticker} 失败: {e}')
"

# 检查代码质量
quality-check: lint test
	@echo "代码质量检查完成"

# 构建发布包
build:
	python setup.py sdist bdist_wheel

# 安装开发环境
install-dev:
	pip install -r requirements.txt
	pip install -e .

# 生成API文档
docs:
	sphinx-build -b html docs/ docs/_build/

# 运行完整的CI流程
ci: format lint test
	@echo "CI流程完成"

# 启动Jupyter Notebook
notebook:
	jupyter notebook notebooks/

# 查看项目统计
stats:
	@echo "项目统计信息:"
	@echo "代码行数:"
	@find src/ -name "*.py" -exec wc -l {} + | tail -1
	@echo "测试文件数:"
	@find tests/ -name "test_*.py" | wc -l
	@echo "模型文件数:"
	@find src/models/ -name "*.py" | wc -l

# 备份项目
backup:
	@DATE=$$(date +%Y%m%d_%H%M%S); \
	tar -czf "backup_stock_prediction_$$DATE.tar.gz" \
		--exclude='.git' \
		--exclude='__pycache__' \
		--exclude='*.pyc' \
		--exclude='data/' \
		--exclude='outputs/' \
		--exclude='logs/' \
		--exclude='cache/' \
		.
	@echo "项目已备份到 backup_stock_prediction_$$(date +%Y%m%d_%H%M%S).tar.gz" 