# 赛事官方原始数据

本目录保存比赛提供的不可变原始数据快照，不进入 Git。

- `dongwu_hk_ipo_2020_2025/`：东吴证券赛道提供的 2020—2025 年港股 IPO
  招股书、公司信息、证券信息、日行情及数据字典。
- 原始 CSV 使用 `GB18030` 编码；项目生成的标准 JSON、JSONL 和 CSV 统一使用
  UTF-8。
- 年度 ZIP 与已解压 PDF 属于同一批资料，导入器只读取已解压 PDF，ZIP 仅作为
  原始交付归档保留。
- `hksharedescription.csv` 当前交付文件在 256 KiB 处截断，不能作为正式证券主表；
  导入器会记录质量问题并从日行情首条记录恢复可验证的上市日期和发行价代理值。

禁止手工修改本目录中的官方文件。标准化公司资料位于 `data/raw/company_info`、
`data/raw/prospectus` 和 `data/raw/market`，建模标签位于 `data/processed/competition`。
