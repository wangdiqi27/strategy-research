# main.py
import importlib.util
import sys
import traceback
import os

from strategy_research.tool.class_loader import load_class_from_file


def run_versions(versions):
    """
    依次运行指定版本列表的研究脚本

    Args:
        versions: 版本号列表，如 [27, 28] 或 [27, 28, 31]
        data:     传给 TrixVptResearch 构造函数的公共数据
    """
    results = {}

    for version in versions:
        file_path = f'trix_vpt_research_v{version}.py'      # 由版本号构造路径
        module_name = f'trix_vpt_research_v{version}'       # 唯一模块名

        try:
            # 加载类（含文件存在性检查）
            cls = load_class_from_file(file_path, module_name)

            # 实例化并执行
            research = cls(data)
            print(f"v{version} 完成")

        except Exception as e:
            print(f"v{version} ❌ 失败: {e}")
            traceback.print_exc()
            exit(0)

        finally:
            # 无论成败，跑完就从 sys.modules 清掉，保持干净
            sys.modules.pop(module_name, None)

    return results

def main():
    # ===== 指定要运行的版本列表 =====
    versions = ["27", "28"]          # ← 想跑哪些就写哪些，如 [27, 28, 30, 31]
    results = run_versions(versions)


if __name__ == '__main__':
    main()
