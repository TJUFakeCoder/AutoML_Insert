# AutoML_Insert
A tool to insert AutoML statements automatically
## 所需依赖
+ redbaron(AST解析)
+ pycg(Call Graph)
## 使用方法
使用前修改项目所在文件夹：
+ line 744:项目文件夹所在文件夹
+ line 745:项目文件夹名
+ line 749:项目函数入口
+ line 757:同744

python main.py --Model (模型框架（pbt、pytorch、keras）) --intermediate_necessary （是否需要中间结果上传（true/false)) --arg_to_report (需要上传的变量名) --params_to_update （需要更新的参数名）
