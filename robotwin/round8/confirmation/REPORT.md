# Round8 paired confirmation and gate ablation

Primary endpoint: fresh query_gate200 vs query_only200; lower errors are better.
Offline errors do not establish closed-loop success.

## fresh: 32 episodes / 64 noise pairs

|Variant|Position cm|Rotation deg|Gripper MAE|
|---|---:|---:|---:|
|initial|0.230886|0.721033|0.0014078|
|query_only|0.231620|0.719202|0.0014087|
|query_gate|0.231545|0.722239|0.0014102|
|gate_off|0.231030|0.721689|0.0014122|
|base|0.298520|0.782623|0.0028599|

query_gate_vs_query_only: normalized excess 95%CI [-0.0027187665148112176, 0.0058446064226473134]; ratios {'position_cm': 0.9996768251397476, 'rotation_deg': 1.0042233067129873, 'gripper_mae': 1.0010526172288585}; improvement supported False.

query_gate_vs_gate_off: normalized excess 95%CI [-0.004180768127866786, 0.004439182728692205]; ratios {'position_cm': 1.0022282162910003, 'rotation_deg': 1.0007617052273206, 'gripper_mae': 0.9985655526956008}; improvement supported False.

query_gate_vs_initial: normalized excess 95%CI [-0.0019296528130394509, 0.0060977467455643865]; ratios {'position_cm': 1.0028565528109294, 'rotation_deg': 1.0016729363549808, 'gripper_mae': 1.0017129135281664}; improvement supported False.

query_only_vs_initial: normalized excess 95%CI [-0.003676967064625667, 0.004572515317545475]; ratios {'position_cm': 1.003180755611432, 'rotation_deg': 0.9974603553403333, 'gripper_mae': 1.0006596019908882}; improvement supported False.

Same-checkpoint gate output drift MSE: 1.1440431855191946e-06.

## tail: 8 episodes / 64 noise pairs

|Variant|Position cm|Rotation deg|Gripper MAE|
|---|---:|---:|---:|
|initial|0.213574|0.492067|0.0016773|
|query_only|0.211977|0.489203|0.0016816|
|query_gate|0.214835|0.493041|0.0016847|
|gate_off|0.216119|0.488549|0.0016740|
|base|0.294915|0.560323|0.0031988|

query_gate_vs_query_only: normalized excess 95%CI [-0.002802607368712281, 0.01800042583735489]; ratios {'position_cm': 1.0134851124937378, 'rotation_deg': 1.0078460268371625, 'gripper_mae': 1.001830384328422}; improvement supported False.

query_gate_vs_gate_off: normalized excess 95%CI [-0.004130256758730715, 0.012778576275114325]; ratios {'position_cm': 0.9940601132898635, 'rotation_deg': 1.0091948065837322, 'gripper_mae': 1.006400336199113}; improvement supported False.

query_gate_vs_initial: normalized excess 95%CI [-0.002387236978672147, 0.011430444966690552]; ratios {'position_cm': 1.005903698155895, 'rotation_deg': 1.0019798280995218, 'gripper_mae': 1.0043753634494492}; improvement supported False.

query_only_vs_initial: normalized excess 95%CI [-0.0131901425659204, 0.005579061849090363]; ratios {'position_cm': 0.9925194615644741, 'rotation_deg': 0.994179469302419, 'gripper_mae': 1.0025403293420108}; improvement supported False.

Same-checkpoint gate output drift MSE: 1.1991665349597724e-06.

## tail_remaining_1: 8 episodes / 16 noise pairs

|Variant|Position cm|Rotation deg|Gripper MAE|
|---|---:|---:|---:|
|initial|0.217197|0.451896|0.0002542|
|query_only|0.209581|0.429582|0.0002845|
|query_gate|0.221765|0.442836|0.0002896|
|gate_off|0.228551|0.430476|0.0002422|
|base|0.341235|0.456511|0.0017360|

query_gate_vs_query_only: normalized excess 95%CI [-0.055047018177230254, 0.11868472000047901]; ratios {'position_cm': 1.0581370669474048, 'rotation_deg': 1.030852163568608, 'gripper_mae': 1.0179979049364933}; improvement supported False.

query_gate_vs_gate_off: normalized excess 95%CI [-0.021413799623722602, 0.14727106465691137]; ratios {'position_cm': 0.9703085056741149, 'rotation_deg': 1.0287123540265257, 'gripper_mae': 1.1957932784741983}; improvement supported False.

query_gate_vs_initial: normalized excess 95%CI [-0.05859605059321858, 0.1502542388938436]; ratios {'position_cm': 1.0210317225609393, 'rotation_deg': 0.9799505465215486, 'gripper_mae': 1.1391469347087093}; improvement supported False.

query_only_vs_initial: normalized excess 95%CI [-0.08329137631494649, 0.10885494206415693]; ratios {'position_cm': 0.9649333290123653, 'rotation_deg': 0.9506218070388986, 'gripper_mae': 1.1190071503926855}; improvement supported False.

Same-checkpoint gate output drift MSE: 1.233477313178355e-06.

## tail_remaining_8: 8 episodes / 16 noise pairs

|Variant|Position cm|Rotation deg|Gripper MAE|
|---|---:|---:|---:|
|initial|0.190157|0.427739|0.0012500|
|query_only|0.191456|0.434555|0.0012589|
|query_gate|0.188538|0.436731|0.0012329|
|gate_off|0.188279|0.436546|0.0012601|
|base|0.249749|0.552989|0.0028071|

query_gate_vs_query_only: normalized excess 95%CI [-0.020511662760716615, -0.0009337909544194074]; ratios {'position_cm': 0.9847594535286471, 'rotation_deg': 1.005008916168165, 'gripper_mae': 0.9794055976454874}; improvement supported True.

query_gate_vs_gate_off: normalized excess 95%CI [-0.023599442384030653, 0.012632514018337013]; ratios {'position_cm': 1.0013758109201734, 'rotation_deg': 1.0004234474451301, 'gripper_mae': 0.9784148429757197}; improvement supported False.

query_gate_vs_initial: normalized excess 95%CI [-0.022135233158269436, 0.01860644529999711]; ratios {'position_cm': 0.9914893492788656, 'rotation_deg': 1.0210237973027372, 'gripper_mae': 0.9863910326106697}; improvement supported False.

query_only_vs_initial: normalized excess 95%CI [-0.014990073090658052, 0.03380138070092789]; ratios {'position_cm': 1.0068340504131272, 'rotation_deg': 1.0159350637361833, 'gripper_mae': 1.007132320850499}; improvement supported False.

Same-checkpoint gate output drift MSE: 1.2023890931800452e-06.

## tail_remaining_24: 8 episodes / 16 noise pairs

|Variant|Position cm|Rotation deg|Gripper MAE|
|---|---:|---:|---:|
|initial|0.215003|0.466579|0.0004867|
|query_only|0.216284|0.469867|0.0004635|
|query_gate|0.216722|0.473756|0.0004816|
|gate_off|0.215725|0.467439|0.0004645|
|base|0.275830|0.542431|0.0020453|

query_gate_vs_query_only: normalized excess 95%CI [0.0031672978814089535, 0.029051138202543143]; ratios {'position_cm': 1.0020237466127566, 'rotation_deg': 1.0082778747385255, 'gripper_mae': 1.039146688890397}; improvement supported False.

query_gate_vs_gate_off: normalized excess 95%CI [0.008208045540749245, 0.028926053441704586]; ratios {'position_cm': 1.004619826270217, 'rotation_deg': 1.0135150207057861, 'gripper_mae': 1.0369505180765617}; improvement supported False.

query_gate_vs_initial: normalized excess 95%CI [-0.012884806223470543, 0.02237025524093085]; ratios {'position_cm': 1.0079944112566819, 'rotation_deg': 1.0153822131668664, 'gripper_mae': 0.9895038888709334}; improvement supported False.

query_only_vs_initial: normalized excess 95%CI [-0.021237780183811762, -0.0021935076065466774]; ratios {'position_cm': 1.005958605935347, 'rotation_deg': 1.007046012420121, 'gripper_mae': 0.9522273413848121}; improvement supported True.

Same-checkpoint gate output drift MSE: 1.2097557799961578e-06.

## tail_remaining_36: 8 episodes / 16 noise pairs

|Variant|Position cm|Rotation deg|Gripper MAE|
|---|---:|---:|---:|
|initial|0.231941|0.622055|0.0047184|
|query_only|0.230587|0.622808|0.0047195|
|query_gate|0.232316|0.618842|0.0047345|
|gate_off|0.231921|0.619736|0.0047290|
|base|0.312846|0.689362|0.0062066|

query_gate_vs_query_only: normalized excess 95%CI [-0.003669095818504105, 0.00681308058553271]; ratios {'position_cm': 1.007502210839738, 'rotation_deg': 0.9936312885160916, 'gripper_mae': 1.0031726169577246}; improvement supported False.

query_gate_vs_gate_off: normalized excess 95%CI [-0.003325638475974897, 0.004633474644331359]; ratios {'position_cm': 1.0017052559772752, 'rotation_deg': 0.9985577534763634, 'gripper_mae': 1.0011572788251106}; improvement supported False.

query_gate_vs_initial: normalized excess 95%CI [-0.006290701632626097, 0.006680245711219682]; ratios {'position_cm': 1.0016168873878815, 'rotation_deg': 0.9948354814733724, 'gripper_mae': 1.0034117571084797}; improvement supported False.

query_only_vs_initial: normalized excess 95%CI [-0.007054208936992084, 0.004585761486809521]; ratios {'position_cm': 0.9941585007074565, 'rotation_deg': 1.001211911270507, 'gripper_mae': 1.0002383838501099}; improvement supported False.

Same-checkpoint gate output drift MSE: 1.1510439534845318e-06.
