import { defineConfig } from 'vitepress'

const isEdgeOne = process.env.EDGEONE === '1'
const baseConfig = isEdgeOne ? '/' : '/signal-to-intelligence/'
const githubRepo = 'https://github.com/datawhalechina/signal-to-intelligence'
const githubCodeBase = `${githubRepo}/blob/main/code/`
const notebookMap: Record<string, string> = {
  '01_dsp': 'ch01_dsp_interactive.ipynb',
  '02_optimization': 'ch02_optimization_interactive.ipynb',
  '03_deep_learning_fast': 'ch03_deep_learning_interactive.ipynb',
  '04_transformer': 'ch04_transformer_interactive.ipynb',
  '05_llm_basics': 'ch05_llm_interactive.ipynb',
  '06_llm_applications': 'ch06_llm_applications_interactive.ipynb',
  '07_multimodal_llm': 'ch07_multimodal_interactive.ipynb',
  '08_llm_engineering': 'ch08_engineering_interactive.ipynb',
}

export default defineConfig({
  lang: 'zh-CN',
  title: "从信号到智能",
  description: "深度学习与大语言模型的完整学习路径",
  base: baseConfig,
  ignoreDeadLinks: true,
  markdown: {
    math: true,
    config(md) {
      md.core.ruler.after('block', 'inject-notebook-block', (state) => {
        const pagePath = String((state.env as { path?: string } | undefined)?.path ?? '')
        if (!/\/docs\/0[1-8]_[^/]+\//.test(pagePath)) return
        if (/\/README\.md$/.test(pagePath)) return
        if (state.src.includes('## 在线 Notebook')) return

        const chapterMatch = pagePath.match(/\/docs\/(0[1-8]_[^/]+)\//)
        const chapter = chapterMatch?.[1]
        const notebookFile = chapter ? notebookMap[chapter] : undefined
        if (!notebookFile) return

        const notebookBlock = `
<section class="notebook-entry">
<p class="notebook-entry-title">在线 Notebook</p>
        <p>对应的交互式版本可在 <a href="https://colab.research.google.com/github/datawhalechina/signal-to-intelligence/blob/main/notebooks/${notebookFile}">Google Colab</a> 打开，统一使用方式见 <a href="/signal-to-intelligence/00_introduction/05_how_to_use_this_tutorial.html">第0章说明</a>。</p>
</section>
`

        const Token = state.tokens[0]?.constructor
        if (!Token) return
        const token = new Token('html_block', '', 0)
        token.content = notebookBlock

        let insertIndex = 0
        const h1OpenIndex = state.tokens.findIndex(
          (item) => item.type === 'heading_open' && item.tag === 'h1'
        )

        if (h1OpenIndex >= 0) {
          const h1CloseIndex = state.tokens.findIndex(
            (item, index) => index > h1OpenIndex && item.type === 'heading_close' && item.tag === 'h1'
          )
          insertIndex = h1CloseIndex >= 0 ? h1CloseIndex + 1 : h1OpenIndex + 1

          while (
            state.tokens[insertIndex]?.type === 'paragraph_open' &&
            state.tokens[insertIndex + 1]?.type === 'inline' &&
            /^\*\*(核心问题|版本|最后更新)：/.test(state.tokens[insertIndex + 1].content.trim())
          ) {
            insertIndex += 3
          }
        }

        state.tokens.splice(insertIndex, 0, token)
      })

      md.core.ruler.after('inline', 'rewrite-code-links', (state) => {
        const walk = (tokens: any[]) => {
          for (const token of tokens) {
            if (token.type === 'link_open') {
              const href = token.attrGet('href')
              if (href && /^(?:\.\.\/)+code\//.test(href)) {
                token.attrSet('href', href.replace(/^(?:\.\.\/)+code\//, githubCodeBase))
              }
            }

            if (token.children && token.children.length > 0) {
              walk(token.children)
            }
          }
        }

        walk(state.tokens)
      })
    }
  },
  themeConfig: {
    logo: '/datawhale-logo.png',
    nav: [
      { text: '代码库', link: 'https://github.com/datawhalechina/signal-to-intelligence' },
    ],
    search: {
      provider: 'local',
      options: {
        translations: {
          button: {
            buttonText: '搜索文档',
            buttonAriaLabel: '搜索文档'
          },
          modal: {
            noResultsText: '无法找到相关结果',
            resetButtonTitle: '清除查询条件',
            footer: {
              selectText: '选择',
              navigateText: '切换'
            }
          }
        }
      }
    },
    sidebar: [
      {
        text: '0. 介绍与学习路径',
        link: '/00_introduction/',
        items: [
          { text: '0.1 为什么是 LLM 时代', link: '/00_introduction/01_why_llm_era' },
          { text: '0.2 学习路径', link: '/00_introduction/02_learning_paths' },
          { text: '0.3 全书概览', link: '/00_introduction/03_overview' },
          { text: '0.4 工具与基础设施', link: '/00_introduction/04_tools_and_infrastructure' },
          { text: '0.5 如何使用本教程', link: '/00_introduction/05_how_to_use_this_tutorial' }
        ]
      },
      {
        text: '1. 数字信号处理基础',
        link: '/01_dsp/',
        items: [
          { text: '1.1 信号的本质', link: '/01_dsp/01_signals' },
          { text: '1.2 傅里叶变换', link: '/01_dsp/02_fourier' },
          { text: '1.3 滤波器设计', link: '/01_dsp/03_filters' },
          { text: '1.4 时频分析', link: '/01_dsp/04_time_freq' },
          { text: '1.5 随机信号', link: '/01_dsp/05_random_signals' },
          { text: '1.6 信号检测', link: '/01_dsp/06_signal_detection' },
          { text: '1.7 信号估计', link: '/01_dsp/07_signal_estimation' },
          { text: '1.8 矩阵分解', link: '/01_dsp/08_matrix_decomposition' },
          {
            text: '深度阅读 (Extensions)',
            items: [
              { text: '1.E1 高级主题', link: '/01_dsp/extensions/advanced_topics' },
              { text: '1.E2 线性系统', link: '/01_dsp/extensions/linear_systems' },
              { text: '1.E3 随机过程', link: '/01_dsp/extensions/stochastic_processes' }
            ]
          }
        ]
      },
      {
        text: '2. 优化算法与传统机器学习',
        link: '/02_optimization/',
        items: [
          { text: '2.1 梯度下降', link: '/02_optimization/01_gradient_descent' },
          { text: '2.2 自适应优化器', link: '/02_optimization/02_adaptive_optimizers' },
          { text: '2.3 优化算法与传统ML', link: '/02_optimization/03_optimization_and_traditional_ml' },
          { text: '2.4 数值方法', link: '/02_optimization/04_numerical_methods' },
          { text: '2.5 线性与逻辑回归', link: '/02_optimization/05_linear_logistic_regression' },
          { text: '2.6 SVM与核方法', link: '/02_optimization/06_svm_kernel_methods' },
          { text: '2.7 决策树与随机森林', link: '/02_optimization/07_decision_trees_random_forest' },
          {
            text: '深度阅读 (Extensions)',
            items: [
              { text: '2.E1 高级优化', link: '/02_optimization/extensions/advanced_optimization' },
              { text: '2.E2 凸分析', link: '/02_optimization/extensions/convex_analysis' },
              { text: '2.E3 运筹学基础', link: '/02_optimization/extensions/operations_research_basics' },
              { text: '2.E4 树数据结构', link: '/02_optimization/extensions/tree_data_structures' }
            ]
          }
        ]
      },
      {
        text: '3. 深度学习快速入门',
        link: '/03_deep_learning_fast/',
        items: [
          { text: '3.1 为什么需要深度学习', link: '/03_deep_learning_fast/01_why_deep_learning' },
          { text: '3.2 CNN 的本质', link: '/03_deep_learning_fast/02_cnn_essence' },
          { text: '3.3 YOLO 目标检测', link: '/03_deep_learning_fast/03_yolo_detection' },
          { text: '3.4 图像分割', link: '/03_deep_learning_fast/04_image_segmentation' },
          { text: '3.5 序列模型与1D信号', link: '/03_deep_learning_fast/05_sequence_models_and_1d_signals' },
          { text: '3.6 为什么Transformer更好', link: '/03_deep_learning_fast/06_why_transformer_better' },
          {
            text: '深度阅读 (Extensions)',
            items: [
              { text: '3.E1 深度学习理论', link: '/03_deep_learning_fast/extensions/deep_learning_theory' }
            ]
          }
        ]
      },
      {
        text: '4. Transformer 架构',
        link: '/04_transformer/',
        items: [
          { text: '4.1 自注意力机制', link: '/04_transformer/01_attention' },
          { text: '4.2 多头注意力', link: '/04_transformer/02_multihead' },
          { text: '4.3 位置编码', link: '/04_transformer/03_positional_encoding' },
          { text: '4.4 Transformer 架构', link: '/04_transformer/04_architecture' },
          {
            text: '深度阅读 (Extensions)',
            items: [
              { text: '4.E1 注意力变体', link: '/04_transformer/extensions/attention_variants' },
              { text: '4.E2 图论基础', link: '/04_transformer/extensions/graph_theory_basics' },
              { text: '4.E3 向量空间基础', link: '/04_transformer/extensions/vector_space_basics' }
            ]
          }
        ]
      },
      {
        text: '5. 大语言模型基础',
        link: '/05_llm_basics/',
        items: [
          { text: '5.1 预训练', link: '/05_llm_basics/01_pretraining' },
          { text: '5.2 训练数据', link: '/05_llm_basics/02_training_data' },
          { text: '5.3 模型家族', link: '/05_llm_basics/03_model_families' },
          { text: '5.4 微调', link: '/05_llm_basics/04_finetuning' },
          { text: '5.5 RL 与对齐', link: '/05_llm_basics/05_rl_alignment' },
          { text: '5.6 模型评估', link: '/05_llm_basics/06_evaluation' },
          {
            text: '深度阅读 (Extensions)',
            items: [
              { text: '5.E1 对齐详解', link: '/05_llm_basics/extensions/alignment_details' },
              { text: '5.E2 评估调查', link: '/05_llm_basics/extensions/evaluation_survey' },
              { text: '5.E3 微调调查', link: '/05_llm_basics/extensions/finetuning_survey' },
              { text: '5.E4 推理优化', link: '/05_llm_basics/extensions/inference_optimization' },
              { text: '5.E5 信息论基础', link: '/05_llm_basics/extensions/information_theory_basics' },
              { text: '5.E6 LLM训练详解', link: '/05_llm_basics/extensions/llm_training_details' }
            ]
          }
        ]
      },
      {
        text: '6. 大语言模型应用',
        link: '/06_llm_applications/',
        items: [
          { text: '6.1 Prompt 工程', link: '/06_llm_applications/01_prompt_engineering' },
          { text: '6.2 微调', link: '/06_llm_applications/02_finetuning' },
          { text: '6.3 RAG', link: '/06_llm_applications/03_rag' },
          { text: '6.4 Agent', link: '/06_llm_applications/04_agent' },
          { text: '6.5 系统设计', link: '/06_llm_applications/05_system_design' },
          {
            text: '深度阅读 (Extensions)',
            items: [
              { text: '6.E1 Agent 高级', link: '/06_llm_applications/extensions/agent_advanced' },
              { text: '6.E2 微调高级', link: '/06_llm_applications/extensions/finetuning_advanced' },
              { text: '6.E3 推理部署', link: '/06_llm_applications/extensions/inference_deployment' },
              { text: '6.E4 RAG 高级', link: '/06_llm_applications/extensions/rag_advanced' }
            ]
          }
        ]
      },
      {
        text: '7. 多模态大语言模型',
        link: '/07_multimodal_llm/',
        items: [
          { text: '7.1 视觉语言融合', link: '/07_multimodal_llm/01_vision_language' },
          { text: '7.2 Qwen VL', link: '/07_multimodal_llm/02_qwen_vl' },
          { text: '7.3 高分辨率图像处理', link: '/07_multimodal_llm/03_high_resolution' },
          { text: '7.4 应用场景', link: '/07_multimodal_llm/04_applications' },
          { text: '7.5 案例研究', link: '/07_multimodal_llm/05_case_studies' },
          {
            text: '深度阅读 (Extensions)',
            items: [
              { text: '7.E1 多模态应用高级', link: '/07_multimodal_llm/extensions/multimodal_applications_advanced' },
              { text: '7.E2 多模态训练详解', link: '/07_multimodal_llm/extensions/multimodal_training_details' }
            ]
          }
        ]
      },
      {
        text: '8. LLM 工程实践与部署',
        link: '/08_llm_engineering/',
        items: [
          { text: '8.1 模型量化与蒸馏', link: '/08_llm_engineering/01_quantization_distillation' },
          { text: '8.2 推理优化', link: '/08_llm_engineering/02_inference_optimization' },
          { text: '8.3 成本优化', link: '/08_llm_engineering/03_cost_optimization' },
          { text: '8.4 评估与基准', link: '/08_llm_engineering/04_evaluation_benchmark' },
          { text: '8.5 生产系统设计', link: '/08_llm_engineering/05_production_system' },
          {
            text: '深度阅读 (Extensions)',
            items: [
              { text: '8.E1 高级量化', link: '/08_llm_engineering/extensions/advanced_quantization' },
              { text: '8.E2 分布式推理', link: '/08_llm_engineering/extensions/distributed_inference' },
              { text: '8.E3 运筹学视角', link: '/08_llm_engineering/extensions/operations_research_serving' },
              { text: '8.E4 推理加速与硬件适配', link: '/08_llm_engineering/extensions/hardware_acceleration_and_conversion' }
            ]
          }
        ]
      },
      {
        text: '附录',
        link: '/appendix/',
        items: [
          { text: 'A 数学参考', link: '/appendix/A_math_reference' },
          { text: 'B 环境配置', link: '/appendix/B_environment_setup' },
          { text: 'C 代码运行指南', link: '/appendix/C_code_guide' },
          { text: 'D 数学基础速查表', link: '/appendix/D_math_quick_reference' },
          { text: 'E 项目维护说明', link: '/appendix/E_project_maintenance' }
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/datawhalechina/signal-to-intelligence' }
    ],

    editLink: {
      pattern: 'https://github.com/datawhalechina/signal-to-intelligence/blob/main/docs/:path'
    },

    footer: {
      message: '本教程采用 CC BY-NC-SA 4.0 许可协议',
      copyright: 'Copyright © 2024-2026'
    }
  }
})
