"""
生成 ch07_case_studies.png 和 ch07_multimodal_applications.png。
两张图都描述任务、流程和验收证据，不包含内置模型分数。

运行：python code/ch07_multimodal_llm/gen_case_studies_chart.py
"""

def plot_case_studies():
    from case_studies import _save_evaluation_map

    _save_evaluation_map(output_path="assets/ch07_case_studies.png")


def plot_multimodal_applications():
    from multimodal_applications import _save_task_map

    _save_task_map(output_path="assets/ch07_multimodal_applications.png")


if __name__ == "__main__":
    plot_case_studies()
    plot_multimodal_applications()
