# Canada28 Matrix AI V4.0

## 本地运行
```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud 部署
1. 将 `app.py`、`requirements.txt`、`README.md` 放在 GitHub 仓库根目录。
2. 在 Streamlit Community Cloud 选择仓库 `zhanghong1231234-wq/canada28-ai-system`、分支 `main`、主文件 `app.py`。
3. 部署成功后打开网页，在左侧上传历史开奖 Excel/CSV。

## 数据格式
必须包含三列号码：`A`、`B`、`C`（每列为 0–9 整数）。
也支持 `号码1`、`号码2`、`号码3`。`期号`可选；建议按时间从旧到新排列。
系统自动计算和值、大小、单双，并生成 0–27 概率矩阵、近期频率、时间顺序回测和 Excel 报告。

## 重要说明
系统输出的是历史统计和概率参考，不是确定性预测。若开奖独立且公平，历史走势不能可靠决定下一期结果。
