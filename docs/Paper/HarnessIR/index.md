---
title: HarnessIR
date: 2026-10-19
tags:
  - 图像恢复
  - 多模态
  - Agent
---

# HarnessIR: Harnessing Multimodal Foundation Models for Universal Real-World Image Restoration

Gap：先前提出的大量AiO图像恢复模型，都是假设单一情形的退化，对真实世界的数据恢复能力有限

历史工作：有人提出使用agent进行图像恢复：对于一张图像，Agent使用视觉语言模型来评估图像的内容和退化情况，然后规划恢复模型的使用顺序

AgenticIR的gap：强行将耦合退化的图像恢复近似为串行的恢复行为；其恢复上限也被视觉语言模型锁死，在恢复前图像的信息就被大量丢弃，只保留了恢复工具的执行序列；与此同时，AgenticIR调用的图像恢复方法本身就是在合成退化数据集上进行训练的，与现实世界图像的分布本身就具有较大差异

作者工作：提出HarnessIR，使用生成式多模态基础模型MFM作为执行器；提出一套工作流：感知诊断、按需调用工具、提示词生成、执行、基于验证的优化；在文章中使用了NanoBanana和GPT-Image-2.5作为执行器

与AgenticIR的区别：HarnessIR的执行器是MFM，而AgenticIR是不同的task-specific恢复方法的调度

Overview：调度闭环基本建立在AgenticIR之上，主要的创新在于构建待恢复图像的证据，加以各种工具辅助（比如分割掩码、深度检测、OCR等），诊断的结果将会以提示词指令形式发送给MFM
