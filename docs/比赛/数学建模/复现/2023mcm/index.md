## 题目描述

### Background

Kenya’s wildlife preserves were originally created primarily to protect wildlife and other natural resources. Kenya’s parliament passed the Wildlife Conservation and Management Act, 2013 to 
provide more equitable sharing of resources, as well as to allow alternative, community-based management efforts[1]. Kenya has since added amendments to address gaps in the legislation to provide more clear governance, finance, and penalties for violators[2].

### Requirement

Focusing on one large game preserve, the Maasai Mara, your task is to determine alternate ways to manage the resources within and outside the current boundaries of the park. Specifically, you should:

- Consider and recommend specific policies and management strategies for different areas within the current preserve that will protect wildlife and other natural resources while also balancing the interests of the people who live in the area. These policies and 
strategies should help mitigate the impacts of lost opportunities experienced by the people who live near the preserve, as well as minimize negative interactions between animals and the people attracted to the preserve.

- Develop and describe a methodology to determine which policies and management strategies will result in the best outcomes. Your report should discuss how to rank and compare outcomes from your methodology. Be sure to include descriptions and analyses 
of the models used to predict the interactions between animals and people, as well as the resulting economic impacts in the area within and around the preserve. 
- Given your proposed plan, provide predictions about the long-term trends that will result 
from your recommendations. Analyze and provide estimates of the certainties and impacts of the possible long-term outcomes. You should also describe how your approach could be applied to other wildlife management areas.

- Finally, provide a two-page non-technical report for the Kenyan Tourism and Wildlife 

Committee discussing your proposed plan and its value for the preserve.

### References

1. The Wildlife Conservation and Management Act, 2013. Republic of Kenya, Kenya Gazette 
Supplement No. 181 (Acts No. 47), 2013.
2. The Wildlife Conservation and Management Act (No. 47 of 2013). Legal Notice No. 155.
Republic of Kenya, Kenya Gazette Supplement No 141 (Acts No. 47), 2020.

## 题目描述（译）

### 背景

肯尼亚野生动物保护区最初的建立主要是为了保护野生动物和其他自然资源。肯尼亚议会于 2013 年通过了《野生动物保护和管理法案》，以提供更公平的资源共享，并允许采取替代性的、基于社区的管理工作[1]。此后，肯尼亚增加了修正案，以解决立法中的漏洞，为违法者提供更明确的治理、财务和处罚[2]。

### 要求

专注于马赛马拉这一大型野生动物保护区，您的任务是确定管理公园当前边界内外资源的替代方法。具体来说，您应该：

- 考虑并建议针对当前保护区内**不同区域**的具体政策和管理策略，以保护野生动物和其他自然资源，同时**平衡该地区居民的利益**。这些政策和战略应有助于**减轻保护区附近居民失去机会的影响**，并**最大限度地减少动物与保护区吸引的人们之间的负面互动**。

- **开发并描述一种方法**来确定哪些政策和管理策略将产生最佳结果。你的报告应该讨论如何对你的方法的结果进行排名和比较。请务必包括对**用于预测动物与人之间相互作用的模型的描述和分析**，以及**对保护区内和周边地区产生的经济影响**。

- 根据您提出的计划，对您的建议所产生的**长期趋势进行预测**。分析并估计可能的长期结果的确定性和影响。您还应该**描述如何将您的方法应用于其他野生动物管理领域**。

- 最后，为肯尼亚旅游和野生动物局提供两页的**非技术性报告**

委员会讨论您提出的计划及其对保护区的价值。

### 参考

《野生动物保护和管理法》，2013年。肯尼亚共和国，肯尼亚公报补充第181号（第47号法案），2013年。

《野生动物保护和管理法》（2013 年第 47 号）。第155号法律公告。肯尼亚共和国，肯尼亚公报补充第141号（第 47 号法案），2020年。

## 初步分析

目标：确定管理公园当前边界内外的资源的**方案**

首先考虑收集**畜牧动物的分布面积**，**野生犀牛的分布面积**，耕地面积，森林面积，**受动物袭击的伤亡人数**等指标的相关数据.其次查找**相关的政府政策**以及初步分析**各政策对不同指标的影响**。**分析指标之间的相关性**(可以采用相关系数或灰色关联度，相关系数需要做假设检验)，根据分析结果可以初步做一些定性分析。

再将指标划分成与动物利益相关的指标和与人的利益相关指标，分析动物利益相关指标和人利益相关指标的函数关系(这里可以采用回归分析)。最后可以可以采用两种思路，第一种对指标进行灵敏度分析，根据灵敏度分析的结果确定最优的政策，第二种将已有政策量化成对指标的影响，再考虑根据模型的计算结果确定最优的政策

与**人类**利益相关指标

- 人均收入
- 生态旅游收入
- 动物与人的“负面互动”
- etc.

与**动物**利益相关指标

- 动物生存区域面积
- 物种多样性
- etc.