# Canada28 Matrix AI V4.2

真实历史数据统计、动态窗口矩阵、极值区间统计和时间顺序回测。

## 本地运行
```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud 部署
1. 将 `app.py`、`requirements.txt`、`README.md` 放在 GitHub 仓库根目录。
2. 在 Streamlit Community Cloud 选择仓库 `zhanghong1231234-wq/canada28-ai-system`、分支 `main`、主文件 `app.py`。
3. 部署成功后，在左侧上传历史开奖 Excel/CSV。

## 数据格式
必须包含三列号码：`A`、`B`、`C`（每列为 0–9 整数）。也支持 `号码1`、`号码2`、`号码3`。`期号`可选；建议按时间从旧到新排列。

## V4.2 回测说明
回测按时间顺序使用前 80% 记录作为起始历史，对后 20% 逐期测试。每一期预测只使用该期开奖之前的数据。单一和值预测取平滑近期概率最高的一个和值，预测和值与实际和值完全相同才计为命中；页面显示命中次数、命中率、未命中次数和逐期明细，并保留 Log Loss、Brier Score。

## 重要说明
回测命中率只描述当前测试样本，不代表下一期命中概率。公平且独立的随机开奖无法由历史数据可靠预测；本工具用于历史统计、概率教学和模型评估，不构成投注建议。
