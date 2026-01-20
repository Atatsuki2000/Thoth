# n8n Integration Workflows

这个文件夹包含预配置的 n8n workflow 模板，可直接导入 n8n 使用 RAG Agent 系统。

## 📋 可用 Workflows

### 1. `rag-query-workflow.json` - RAG 查询
**功能：** 查询知识库并获取 AI 生成的答案

**节点：**
- Manual Trigger（手动触发）
- HTTP Request（调用 KB API `/query`）
- Code（格式化响应）

**使用方法：**
1. 在 n8n 中点击 "Import from File"
2. 选择此文件
3. 修改查询参数（query, collection_name, top_k, min_similarity）
4. 点击 "Execute Workflow"

**示例输出：**
```json
{
  "query": "What is machine learning?",
  "answer": "Machine learning is...",
  "documentCount": 3,
  "documents": [...]
}
```

---

### 2. `kb-upload-workflow.json` - 文档上传
**功能：** 将文档上传到知识库

**节点：**
- Manual Trigger
- Read Binary File（读取本地文件）
- HTTP Request（上传到 `/upload`）
- Code（格式化结果）

**使用方法：**
1. 导入 workflow
2. 在 "Read Document File" 节点设置文件路径
3. 修改 collection_name（可选）
4. 执行 workflow

**支持格式：** PDF, TXT, MD, DOCX

---

### 3. `automated-rag-workflow.json` - 自动化知识助手
**功能：** 定时自动查询知识库并生成日报

**节点：**
- Schedule Trigger（每天早上 9 点）
- Code（准备问题列表）
- HTTP Request（批量查询）
- Code（编译报告）
- Set（输出）

**使用方法：**
1. 导入 workflow
2. 在 "Prepare Questions" 节点自定义问题列表
3. 调整 cron 表达式更改执行时间
4. 激活 workflow（Active: ON）

**Cron 示例：**
- `0 9 * * *` - 每天 9:00
- `0 */6 * * *` - 每 6 小时
- `0 9 * * 1` - 每周一 9:00

---

## 🚀 快速开始

### 前置条件
1. 已安装 n8n：`npm install -g n8n`
2. RAG 系统正在运行：`.\start_kb_system.ps1`
3. KB API 可访问：`http://localhost:8100`

### 导入步骤
1. 启动 n8n：`n8n start`
2. 访问：`http://localhost:5678`
3. 点击右上角 **"Import from File"**
4. 选择 workflow JSON 文件
5. 点击 **"Save"** 保存
6. 点击 **"Execute Workflow"** 测试

---

## 🔧 自定义配置

### 修改 API 端点
如果 KB API 运行在不同端口，在 HTTP Request 节点中修改 URL：
```
http://localhost:8100/query  →  http://your-host:port/query
```

### 添加身份验证
如果启用了 API 认证，在 HTTP Request 节点添加 Headers：
```json
{
  "Authorization": "Bearer YOUR_API_KEY"
}
```

### 连接其他服务
可以在 workflow 末尾添加节点：
- **Slack** - 发送报告到频道
- **Email** - 邮件通知
- **Google Sheets** - 保存结果
- **Webhook** - 触发其他自动化

---

## 📚 更多资源

- [n8n 官方文档](https://docs.n8n.io/)
- [HTTP Request 节点](https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.httprequest/)
- [Cron 表达式生成器](https://crontab.guru/)

## 💡 提示

- 使用 n8n 的 **Variables** 功能存储 API URL 和 collection 名称
- 启用 workflow 的 **Error Workflow** 处理失败情况
- 使用 **Sticky Notes** 在 workflow 中添加注释
