# 中间数据

保存可重建的页面文本、OCR结果、切片、坐标和临时索引。

```text
data/interim/
├── documents/<公司目录>/<document_id>/pages.jsonl
├── chunks/<公司目录>/<document_id>/chunks.jsonl
└── embeddings/<公司目录>/<document_id>/<embedding_model>/
    ├── vectors.npy
    ├── metadata.json
    └── manifest.json
```

页面和切片由 `scripts/analyze_prospectus.py` 自动重建；向量在首次语义检索时生成，模型、
维度、源哈希或切片文本变化后自动失效并重建。
这些文件都不作为原始事实来源，实际版本身份由 `data/app.db` 与源文件SHA-256确定。
