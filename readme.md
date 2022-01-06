
## 训练流程模板

- def train() 进行指标数据的更新和保存
- logger_hook负责在合适的时间拿到这些指标数值进行打印


- 基本流程
    - 添加eval_hook
    - 添加early_stop_hook
    - 修改checkpoint_hook
    - logger丰富优化
- 数据类型
- 文档注释
- 扩展


### hook基本信息
- 每 eval_period进行一次验证集计算

- logger_hook:
    - 每 eval_period进行一次logger
    - 每 epoch后进行一次logger

- checkpoint_hook
    - 每 eval_period进行一次checkpoint判断，有提高，保存
    - 每 epoch后进行一次checkpoint判断，有提高，保存

- early_stop_hook:
    - 每 eval_period进行一次判断，当前epoch有没有提高
    - 每 epoch后判断当前epoch有没有提高，是否需要早停