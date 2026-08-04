# 新版分析工作台

新版界面使用 React + TypeScript，后端继续复用项目现有的 Python、LangGraph、
PostgreSQL/pgvector 和批量分析任务。

## 开发模式

在项目 Conda 环境中启动 API：

```powershell
python scripts/run_web_api.py --reload
```

另开一个终端启动前端：

```powershell
cd frontend
npm install
npm run dev
```

访问 `http://127.0.0.1:5173`。

`npm run dev` 或 `npm run preview` 只负责前端页面，仍需同时启动上面的 Python 后端；页面右上角会显示真实连接状态。

## 单服务运行

构建前端后，FastAPI会直接托管网页：

```powershell
cd frontend
npm run build
cd ..
python scripts/run_web_api.py
```

访问 `http://127.0.0.1:8000`。

也可以在 `frontend` 目录执行 `npm run start`，它会使用项目的 Python 环境启动统一服务。正式使用推荐访问 `8000`，避免只启动前端而遗漏 API。
