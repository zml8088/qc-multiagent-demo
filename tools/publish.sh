#!/usr/bin/env bash
# 一键把本地仓库推送到 GitHub / Gitee，并把仓库地址回填到文档占位符。
#
# 用法（在项目根目录执行）：
#   bash tools/publish.sh <GitHub用户名> [仓库名] [Gitee用户名]
#
# 例：
#   bash tools/publish.sh zhangsan                       # 只推 GitHub，仓库名默认 qc-multiagent-demo
#   bash tools/publish.sh zhangsan qc-multiagent-demo zhangsan   # GitHub + Gitee 都推
#
# 前置条件：
#   1) 已在平台网页上新建好「空仓库」（不要勾选 README / .gitignore / license）
#   2) 已准备好令牌：
#      GitHub  Settings -> Developer settings -> Personal access tokens -> Tokens (classic) -> 勾选 repo
#      Gitee   设置 -> 私人令牌 -> 勾选 projects
#      推送时用户名填你的账号名，密码框里粘贴令牌（不是登录密码）
#
# 令牌不会被本脚本读取或保存，全部由 git 在终端交互式询问。

set -euo pipefail

GH_USER="${1:-}"
REPO="${2:-qc-multiagent-demo}"
GITEE_USER="${3:-}"

if [ -z "$GH_USER" ]; then
  echo "用法: bash tools/publish.sh <GitHub用户名> [仓库名] [Gitee用户名]"
  exit 1
fi

# 必须在项目根目录
if [ ! -f "app.py" ] || [ ! -f "README.md" ]; then
  echo "请在项目根目录 qc-multiagent-demo/ 下执行本脚本"
  exit 1
fi

# 统一分支名为 main（Gitee 也认 main，不必再切 master）
git branch -M main

# ---- GitHub ----
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "https://github.com/${GH_USER}/${REPO}.git"
else
  git remote add origin "https://github.com/${GH_USER}/${REPO}.git"
fi
echo "==> 推送到 GitHub: https://github.com/${GH_USER}/${REPO}.git"
git push -u origin main

# ---- Gitee（可选）----
if [ -n "$GITEE_USER" ]; then
  if git remote get-url gitee >/dev/null 2>&1; then
    git remote set-url gitee "https://gitee.com/${GITEE_USER}/${REPO}.git"
  else
    git remote add gitee "https://gitee.com/${GITEE_USER}/${REPO}.git"
  fi
  echo "==> 推送到 Gitee: https://gitee.com/${GITEE_USER}/${REPO}.git"
  git push -u gitee main
fi

# ---- 回填文档里的占位符 ----
fill() {
  local file="$1"
  [ -f "$file" ] || return 0
  sed -i \
    -e "s|https://github.com/<用户名>/<仓库名>|https://github.com/${GH_USER}/${REPO}|g" \
    -e "s|https://github.com/<你的用户名>/<仓库名>|https://github.com/${GH_USER}/${REPO}|g" \
    -e "s|https://github.com/<用户名>/qc-multiagent-demo|https://github.com/${GH_USER}/${REPO}|g" \
    "$file"
}
fill README.md
fill docs/实践报告.md

echo
echo "==> 推送完成。接下来请在网页上把仓库可见性改为 Public："
echo "    GitHub  ${REPO} -> Settings -> Danger Zone -> Change visibility -> Public"
echo "    Gitee   仓库   -> 管理 -> 仓库设置 -> 开源（公开）"
echo
echo "==> 文档中的仓库地址已回填为 https://github.com/${GH_USER}/${REPO}"
echo "    如需提交这一步的改动：git add -A && git commit -m 'docs: 回填开源仓库地址' && git push"
