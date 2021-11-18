def get_policy_info(policy_name):
    reward_range = [-float('inf'), float('inf')]

    # ----------------------------------- training policies -----------------------------------
    if policy_name == "MultiBandit_5000":  # 1000000 trs
        avg_step, true_value, true_variance = 1.0, 0.7849543071794872, 0.19368630895415684

    elif policy_name == "GridWorld_5000":  # 1000000 trs
        avg_step, true_value, true_variance = 7.432377000000001, 2.924171, 21.451484962759007

    elif policy_name == "CartPole_1000":  # 1000000 trs
        avg_step, true_value, true_variance = 48.483092, 36.045859992477716, 268.52194290384494

    elif policy_name == "CartPoleContinuous_3000":  # 1000000 trs
        avg_step, true_value, true_variance = 49.55973899999999, 35.35066209876531, 385.89637749183777

    # ----------------------------------- handmade policies -----------------------------------
    elif policy_name == "MultiBanditM1S1_0":  # 1000000 trs
        avg_step, true_value, true_variance = 1.0, 0.97801129187375, 0.3243569666114263
    elif policy_name == "MultiBanditM1S1_1":  # 1000000 trs
        avg_step, true_value, true_variance = 1.0, 0.9381294731483403, 0.34066673684870824
    elif policy_name == "MultiBanditM1S1_2":  # 1000000 trs
        avg_step, true_value, true_variance = 1.0, 0.89821351822862, 0.3542639245359842
    elif policy_name == "MultiBanditM1S1_3":  # 1000000 trs
        avg_step, true_value, true_variance = 1.0, 0.8585252227403297, 0.36379649057613617
    elif policy_name == "MultiBanditM1S1_4":  # 1000000 trs
        avg_step, true_value, true_variance = 1.0, 0.8188095477557642, 0.37015771075183906
    elif policy_name == "MultiBanditM1S1_5":  # 1000000 trs
        avg_step, true_value, true_variance = 1.0, 0.7792353239900575, 0.37490084230899906
    elif policy_name == "MultiBanditM1S1_6":  # 1000000 trs
        avg_step, true_value, true_variance = 1.0, 0.7392912613505819, 0.374746953100734
    elif policy_name == "MultiBanditM1S1_7":  # 1000000 trs
        avg_step, true_value, true_variance = 1.0, 0.699676517590132, 0.37245279501101086
    elif policy_name == "MultiBanditM1S1_8":  # 1000000 trs
        avg_step, true_value, true_variance = 1.0, 0.6601875281277058, 0.36617643972159036
    elif policy_name == "MultiBanditM1S1_9":  # 1000000 trs
        avg_step, true_value, true_variance = 1.0, 0.6198911167924824, 0.3575516674618844
    elif policy_name == "MultiBanditM1S1_10":  # 1000000 trs
        avg_step, true_value, true_variance = 1.0, 0.5805710678627818, 0.3449457509023611

    elif policy_name == "MultiBanditM1S1_5000":  # 1000000 trs
        avg_step, true_value, true_variance = 1.0, 0.7846518051421868, 0.1943458642775073
    elif policy_name == "MultiBanditM1S2_5000":  # 1000000 trs
        avg_step, true_value, true_variance = 1.0, 0.7844230758925304, 0.6429200308248614
    elif policy_name == "MultiBanditM1S3_5000":  # 1000000 trs
        avg_step, true_value, true_variance = 1.0, 0.7841943466428745, 1.3905943302960853
    elif policy_name == "MultiBanditM1S4_5000":  # 1000000 trs
        avg_step, true_value, true_variance = 1.0, 0.7839656173932177, 2.43736876269118
    elif policy_name == "MultiBanditM1S5_5000":  # 1000000 trs
        avg_step, true_value, true_variance = 1.0, 0.783736888143561, 3.783243328010147
    elif policy_name == "MultiBanditM1S6_5000":  # 1000000 trs
        avg_step, true_value, true_variance = 1.0, 0.7835081588939052, 5.428218026252981
    elif policy_name == "MultiBanditM1S7_5000":  # 1000000 trs
        avg_step, true_value, true_variance = 1.0, 0.7832794296442492, 7.372292857419695
    elif policy_name == "MultiBanditM1S8_5000":  # 1000000 trs
        avg_step, true_value, true_variance = 1.0, 0.7830507003945921, 9.615467821510267
    elif policy_name == "MultiBanditM1S9_5000":  # 1000000 trs
        avg_step, true_value, true_variance = 1.0, 0.7828219711449363, 12.157742918524715
    elif policy_name == "MultiBanditM1S10_5000":  # 1000000 trs
        avg_step, true_value, true_variance = 1.0, 0.78259324189528, 14.999118148463038
    elif policy_name == "MultiBanditM2S1_5000":  # 1000000 trs
        avg_step, true_value, true_variance = 1.0, 1.5695323395340288, 0.32888532340112764
    elif policy_name == "MultiBanditM3S1_5000":  # 1000000 trs
        avg_step, true_value, true_variance = 1.0, 2.354412873925874, 0.5531684438327968
    elif policy_name == "MultiBanditM4S1_5000":  # 1000000 trs
        avg_step, true_value, true_variance = 1.0, 3.139293408317716, 0.8671952255725136
    elif policy_name == "MultiBanditM5S1_5000":  # 1000000 trs
        avg_step, true_value, true_variance = 1.0, 3.9241739427095585, 1.2709656686202795
    elif policy_name == "MultiBanditM6S1_5000":  # 1000000 trs
        avg_step, true_value, true_variance = 1.0, 4.709054477101401, 1.7644797729760926
    elif policy_name == "MultiBanditM7S1_5000":  # 1000000 trs
        avg_step, true_value, true_variance = 1.0, 5.493935011493244, 2.3477375386399553
    elif policy_name == "MultiBanditM8S1_5000":  # 1000000 trs
        avg_step, true_value, true_variance = 1.0, 6.278815545885086, 3.020738965611866
    elif policy_name == "MultiBanditM9S1_5000":  # 1000000 trs
        avg_step, true_value, true_variance = 1.0, 7.063696080276928, 3.783484053891822
    elif policy_name == "MultiBanditM10S1_5000":  # 1000000 trs
        avg_step, true_value, true_variance = 1.0, 7.848576614668771, 4.635972803479833

    elif policy_name == "GridWorldM1S0_0":  # 1000000 trs
        avg_step, true_value, true_variance = 6.0, 7.0, 0.0
    elif policy_name == "GridWorldM1S0_1":  # 1000000 trs
        avg_step, true_value, true_variance = 6.609939, 6.042702, 5.860234539196002
    elif policy_name == "GridWorldM1S0_2":  # 1000000 trs
        avg_step, true_value, true_variance = 7.364935000000001, 4.819749, 14.871830576998992
    elif policy_name == "GridWorldM1S0_3":  # 1000000 trs
        avg_step, true_value, true_variance = 8.333083, 3.237678, 28.527133168315995
    elif policy_name == "GridWorldM1S0_4":  # 1000000 trs
        avg_step, true_value, true_variance = 9.601637, 1.130083, 50.53441941311099
    elif policy_name == "GridWorldM1S0_5":  # 1000000 trs
        avg_step, true_value, true_variance = 11.336105, -1.771352, 88.05112809209596
    elif policy_name == "GridWorldM1S0_6":  # 1000000 trs
        avg_step, true_value, true_variance = 13.825573, -5.927889, 159.55445500367907
    elif policy_name == "GridWorldM1S0_7":  # 1000000 trs
        avg_step, true_value, true_variance = 17.590277, -12.312398, 314.7793234895959
    elif policy_name == "GridWorldM1S0_8":  # 1000000 trs
        avg_step, true_value, true_variance = 23.791308, -22.981371, 715.9638479603593
    elif policy_name == "GridWorldM1S0_9":  # 1000000 trs
        avg_step, true_value, true_variance = 34.381895, -41.94199, 1741.2111548399
    elif policy_name == "GridWorldM1S0_10":  # 1000000 trs
        avg_step, true_value, true_variance = 51.210699, -73.72804, 3448.617717758401

    elif policy_name == "GridWorldM1S1_5000":  # 1000000 trs
        avg_step, true_value, true_variance = 7.43229, 2.9255798294563666, 23.902357493742574
    elif policy_name == "GridWorldM1S2_5000":  # 1000000 trs
        avg_step, true_value, true_variance = 7.43229, 2.9250766589127344, 31.352746924044226
    elif policy_name == "GridWorldM1S3_5000":  # 1000000 trs
        avg_step, true_value, true_variance = 7.43229, 2.9245734883691, 43.755075568015954
    elif policy_name == "GridWorldM1S4_5000":  # 1000000 trs
        avg_step, true_value, true_variance = 7.43229, 2.9240703178254654, 61.10934342565777
    elif policy_name == "GridWorldM1S5_5000":  # 1000000 trs
        avg_step, true_value, true_variance = 7.43229, 2.9235671472818323, 83.41555049696964
    elif policy_name == "GridWorldM1S6_5000":  # 1000000 trs
        avg_step, true_value, true_variance = 7.43229, 2.923063976738198, 110.67369678195158
    elif policy_name == "GridWorldM1S7_5000":  # 1000000 trs
        avg_step, true_value, true_variance = 7.43229, 2.9225608061945656, 142.8837822806037
    elif policy_name == "GridWorldM1S8_5000":  # 1000000 trs
        avg_step, true_value, true_variance = 7.43229, 2.9220576356509325, 180.04580699292572
    elif policy_name == "GridWorldM1S9_5000":  # 1000000 trs
        avg_step, true_value, true_variance = 7.43229, 2.9215544651072975, 222.15977091891804
    elif policy_name == "GridWorldM1S10_5000":  # 1000000 trs
        avg_step, true_value, true_variance = 7.43229, 2.9210512945636653, 269.2256740585804
    elif policy_name == "GridWorldM2S1_5000":  # 1000000 trs
        avg_step, true_value, true_variance = 7.43229, 5.8516628294563615, 88.13655993487204
    elif policy_name == "GridWorldM3S1_5000":  # 1000000 trs
        avg_step, true_value, true_variance = 7.43229, 8.777745829456363, 195.17857693022347
    elif policy_name == "GridWorldM4S1_5000":  # 1000000 trs
        avg_step, true_value, true_variance = 7.43229, 11.703828829456366, 345.0284084797971
    elif policy_name == "GridWorldM5S1_5000":  # 1000000 trs
        avg_step, true_value, true_variance = 7.43229, 14.629911829456367, 537.6860545835929
    elif policy_name == "GridWorldM6S1_5000":  # 1000000 trs
        avg_step, true_value, true_variance = 7.43229, 17.55599482945637, 773.1515152416109
    elif policy_name == "GridWorldM7S1_5000":  # 1000000 trs
        avg_step, true_value, true_variance = 7.43229, 20.48207782945637, 1051.4247904538497
    elif policy_name == "GridWorldM8S1_5000":  # 1000000 trs
        avg_step, true_value, true_variance = 7.43229, 23.408160829456364, 1372.505880220311
    elif policy_name == "GridWorldM9S1_5000":  # 1000000 trs
        avg_step, true_value, true_variance = 7.43229, 26.334243829456355, 1736.394784540995
    elif policy_name == "GridWorldM10S1_5000":  # 1000000 trs
        avg_step, true_value, true_variance = 7.43229, 29.260326829456346, 2143.0915034159


    else:
        raise NotImplementedError(f'Policy {policy_name} not found!')

    exp_name, training_episodes = policy_name.split('_')
    training_episodes = int(training_episodes)  # deprecated

    if 'CartPole' in policy_name:
        reward_range = [1, 1]
    # elif exp_name == 'MountainCar':
    #     reward_range = [-1, -1]

    return exp_name, training_episodes, avg_step, true_value, true_variance, reward_range
