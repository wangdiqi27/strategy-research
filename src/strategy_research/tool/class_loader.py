import importlib.util
import sys


def load_class_from_file(file_path, module_name, class_name='TrixVptResearch'):
    """从指定 python 文件动态加载一个类"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return getattr(module, class_name)