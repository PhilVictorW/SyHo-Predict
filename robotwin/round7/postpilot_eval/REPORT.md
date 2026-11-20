# Query-only pilot: independent offline confirmation

Step 200 is the prespecified primary endpoint. Intermediate checkpoints are exploratory.
Each table averages within episode first; confidence intervals resample episodes.
Lower error is better. These are offline action errors, not task success rates.

## fresh: 32 episodes / 64 noise pairs

|Checkpoint|Position cm|Rotation deg|Gripper MAE|
|---|---:|---:|---:|
|step_000000|0.242568|0.508435|0.0034195|
|step_000050|0.241656|0.509467|0.0034277|
|step_000100|0.241983|0.509588|0.0034345|
|step_000150|0.241520|0.510901|0.0034223|
|step_000200|0.241807|0.509223|0.0034309|
|base|0.308243|0.584441|0.0047202|

Primary ratios vs initialization: {'position_cm': 0.9968657193473064, 'rotation_deg': 1.0015508195256921, 'gripper_mae': 1.0033518805837027}
Normalized excess 95% CI: [-0.0012676658434816503, 0.0022454957276350565]; screen passed: True.

## tail: 8 episodes / 64 noise pairs

|Checkpoint|Position cm|Rotation deg|Gripper MAE|
|---|---:|---:|---:|
|step_000000|0.261803|0.580135|0.0021869|
|step_000050|0.263013|0.576984|0.0022060|
|step_000100|0.262469|0.579272|0.0022110|
|step_000150|0.263287|0.581793|0.0022198|
|step_000200|0.262541|0.578750|0.0021465|
|base|0.358292|0.681420|0.0034937|

Primary ratios vs initialization: {'position_cm': 1.0028205142756612, 'rotation_deg': 0.997612546378047, 'gripper_mae': 0.9815445780598394}
Normalized excess 95% CI: [-0.011168675615673012, 0.003911265669147078]; screen passed: True.

## tail_remaining_1: 8 episodes / 16 noise pairs

|Checkpoint|Position cm|Rotation deg|Gripper MAE|
|---|---:|---:|---:|
|step_000000|0.250931|0.661016|0.0005702|
|step_000050|0.253948|0.659748|0.0006220|
|step_000100|0.253796|0.657928|0.0006516|
|step_000150|0.256069|0.665410|0.0006894|
|step_000200|0.250869|0.657707|0.0003957|
|base|0.369507|0.819965|0.0017539|

Primary ratios vs initialization: {'position_cm': 0.9997515000637096, 'rotation_deg': 0.9949941497186784, 'gripper_mae': 0.694032751755025}
Normalized excess 95% CI: [-0.08298996572537681, 0.014157549571430337]; screen passed: True.

## tail_remaining_8: 8 episodes / 16 noise pairs

|Checkpoint|Position cm|Rotation deg|Gripper MAE|
|---|---:|---:|---:|
|step_000000|0.289087|0.555712|0.0004571|
|step_000050|0.292410|0.545215|0.0004672|
|step_000100|0.290096|0.550947|0.0004486|
|step_000150|0.288638|0.556741|0.0004643|
|step_000200|0.290410|0.549954|0.0004532|
|base|0.379133|0.617010|0.0019353|

Primary ratios vs initialization: {'position_cm': 1.0045747803866563, 'rotation_deg': 0.9896383262673764, 'gripper_mae': 0.9914139919640506}
Normalized excess 95% CI: [-0.008486281210470936, 0.0031974094500237292]; screen passed: True.

## tail_remaining_24: 8 episodes / 16 noise pairs

|Checkpoint|Position cm|Rotation deg|Gripper MAE|
|---|---:|---:|---:|
|step_000000|0.204466|0.613921|0.0010943|
|step_000050|0.202633|0.613257|0.0010924|
|step_000100|0.204447|0.617262|0.0010948|
|step_000150|0.205173|0.615332|0.0010713|
|step_000200|0.205765|0.615071|0.0011037|
|base|0.313391|0.694771|0.0022294|

Primary ratios vs initialization: {'position_cm': 1.0063578928906043, 'rotation_deg': 1.0018731198067632, 'gripper_mae': 1.00853829905526}
Normalized excess 95% CI: [-0.0007571521355365913, 0.007881901798666994]; screen passed: True.

## tail_remaining_36: 8 episodes / 16 noise pairs

|Checkpoint|Position cm|Rotation deg|Gripper MAE|
|---|---:|---:|---:|
|step_000000|0.302726|0.489891|0.0066259|
|step_000050|0.303061|0.489716|0.0066423|
|step_000100|0.301535|0.490952|0.0066489|
|step_000150|0.303267|0.489689|0.0066543|
|step_000200|0.303120|0.492268|0.0066335|
|base|0.371137|0.593934|0.0080562|

Primary ratios vs initialization: {'position_cm': 1.0013000116554194, 'rotation_deg': 1.004851946308958, 'gripper_mae': 1.0011473590757258}
Normalized excess 95% CI: [-0.00223305857845359, 0.006021958192073903]; screen passed: True.

Fresh screen: True; all tail subgroup screens: True.
Passing a non-regression screen does not establish improvement. Tail cohort has only 8 independent episodes.
No full Stage II training or online rollout was started.
