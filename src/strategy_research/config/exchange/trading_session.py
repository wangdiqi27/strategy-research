from datetime import time

DCE_CZCE_NORMAL_DAY_SESSION: list[tuple[time, time]] = [
    (time(9, 0), time(10, 14)),
    (time(10, 30), time(11, 29)),
    (time(13, 30), time(14, 59)),
]

DCE_CZCE_NORMAL_DAY_NIGHT_SESSION: list[tuple[time, time]] = [
    (time(21, 0), time(22, 59)),

    (time(9, 0), time(10, 14)),
    (time(10, 30), time(11, 29)),
    (time(13, 30), time(14, 59)),
]

CFF_NORMAL_DAY_SESSION: list[tuple[time, time]] = [
    (time(9, 30), time(11, 29)),
    (time(13, 00), time(14, 59)),
]

SHFE_AU_AG_SC_SESSION: list[tuple[time, time]] = [
    (time(21, 0), time(23, 59)),
    (time(0, 0), time(2, 29)),

    (time(9, 0), time(10, 14)),
    (time(10, 30), time(11, 29)),
    (time(13, 30), time(14, 59)),
]

SHFE_CU_SESSION: list[tuple[time, time]] = [
    (time(21, 0), time(23, 59)),
    (time(0, 0), time(0, 59)),

    (time(9, 0), time(10, 14)),
    (time(10, 30), time(11, 29)),
    (time(13, 30), time(14, 59)),
]

SHFE_NORMAL_DAY_SESSION: list[tuple[time, time]] = [
    (time(9, 0), time(10, 14)),
    (time(10, 30), time(11, 29)),
    (time(13, 30), time(14, 59)),
]

SHFE_NORMAL_DAY_NIGHT_SESSION: list[tuple[time, time]] = [
    (time(21, 0), time(22, 59)),

    (time(9, 0), time(10, 14)),
    (time(10, 30), time(11, 29)),
    (time(13, 30), time(14, 59)),
]