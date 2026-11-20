# LIBERO Stage III 子任务评测结果

统计口径：每个子任务取三轮评测中的最高成功率，每个子任务评测 50 次。下表精度单位为 %。

## Spatial（libero_spatial）

| 子任务名称 | 精度（%） |
| --- | ---: |
| pick up the black bowl between the plate and the ramekin and place it on the plate | 100 |
| pick up the black bowl next to the ramekin and place it on the plate | 100 |
| pick up the black bowl from table center and place it on the plate | 100 |
| pick up the black bowl on the cookie box and place it on the plate | 100 |
| pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 100 |
| pick up the black bowl on the ramekin and place it on the plate | 100 |
| pick up the black bowl next to the cookie box and place it on the plate | 100 |
| pick up the black bowl on the stove and place it on the plate | 100 |
| pick up the black bowl next to the plate and place it on the plate | 100 |
| pick up the black bowl on the wooden cabinet and place it on the plate | 100 |

**Spatial 精度：100%**

## Object（libero_object）

| 子任务名称 | 精度（%） |
| --- | ---: |
| pick up the alphabet soup and place it in the basket | 100 |
| pick up the cream cheese and place it in the basket | 100 |
| pick up the salad dressing and place it in the basket | 98 |
| pick up the bbq sauce and place it in the basket | 100 |
| pick up the ketchup and place it in the basket | 100 |
| pick up the tomato sauce and place it in the basket | 100 |
| pick up the butter and place it in the basket | 100 |
| pick up the milk and place it in the basket | 100 |
| pick up the chocolate pudding and place it in the basket | 100 |
| pick up the orange juice and place it in the basket | 100 |

**Object 精度：99.8%**

## Goal（libero_goal）

| 子任务名称 | 精度（%） |
| --- | ---: |
| open the middle drawer of the cabinet | 100 |
| put the bowl on the stove | 98 |
| put the wine bottle on top of the cabinet | 96 |
| open the top drawer and put the bowl inside | 100 |
| put the bowl on top of the cabinet | 100 |
| push the plate to the front of the stove | 100 |
| put the cream cheese in the bowl | 100 |
| turn on the stove | 100 |
| put the bowl on the plate | 98 |
| put the wine bottle on the rack | 98 |

**Goal 精度：99%**

## Long（libero_10）

| 子任务名称 | 精度（%） |
| --- | ---: |
| put both the alphabet soup and the tomato sauce in the basket | 98 |
| put both the cream cheese box and the butter in the basket | 98 |
| turn on the stove and put the moka pot on it | 100 |
| put the black bowl in the bottom drawer of the cabinet and close it | 96 |
| put the white mug on the left plate and put the yellow and white mug on the right plate | 100 |
| pick up the book and place it in the back compartment of the caddy | 100 |
| put the white mug on the plate and put the chocolate pudding to the right of the plate | 96 |
| put both the alphabet soup and the cream cheese box in the basket | 100 |
| put both moka pots on the stove | 96 |
| put the yellow and white mug in the microwave and close it | 94 |

**Long 精度：97.8%**

**综合精度：99.15%**

数据来源：[taskwise_best_of_three.csv](stage_III/result/libero/full40x50_20261014_201831/taskwise_best_of_three.csv)。
