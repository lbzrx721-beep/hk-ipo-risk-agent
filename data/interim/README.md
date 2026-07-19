# 中间数据

保存可重建的页面文本、OCR结果、切片、坐标和临时索引。

```text
data/interim/
├── documents/<公司目录>/<document_id>/pages.jsonl
├── chunks/<公司目录>/<document_id>/chunks.jsonl
└── embeddings/<公司目录>/<document_id>/<embedding_model>/
    ├── embeddings.npy
    ├── metadata.jsonl
    └── manifest.json
```

页面和切片由 `scripts/analyze_prospectus.py` 自动重建；向量将在语义检索阶段生成。
这些文件都不作为原始事实来源，实际版本身份由 `data/app.db` 与源文件SHA-256确定。
