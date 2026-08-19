# Aelios 仓库操作规则

## 生产部署

- Aelios 的生产部署只允许通过将用户已批准的修改提交并推送到 GitHub 仓库 `oops33333/Aelios` 的 `main` 分支，由已连接的 Cloudflare 构建与部署流程触发。
- 禁止在本地或由代理直接运行 `wrangler deploy`、`npm run deploy`、`npm run deploy:cloudflare`，以及任何等价的 Cloudflare 直接部署命令。
- `git commit ...` 与 `git push ...` 是两个独立操作，分别涉及 Git 写入和网络、远端写入。每一步都必须先向用户展示完整的单条命令，并等待用户明确回复“同意”后才能执行；不得将两步拼接成一条命令。
- 不得读取、显示、复制或搜索 GitHub、Cloudflare 凭据、访问令牌、API key、密码或其他秘密。
