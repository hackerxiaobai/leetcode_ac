from typing import List, Dict, Tuple

class Solution:
    def readBinaryWatch(self, num: int) -> List[str]:
    	'''
		desc: 二进制手表
		二进制手表顶部有 4 个 LED 代表 小时（0-11），底部的 6 个 LED 代表 分钟（0-59）。
		8 4 2 1
		32 16 8 4 2 1
		给定一个非负整数 n 代表当前 LED 亮着的数量，返回所有可能的时间。
    	'''
    	res = []
        hour_seen = set()
        minute_seen = set()
        def backtrace(num: int, hour: int, minute: int, which: int) -> None:
            if hour > 11 or minute > 59:
                return
            if num == 0:
                res.append(f'{hour}:{minute:02}')
                return
            # 枚举 hour 可能，对应 h 是 2 ^ h
            for h in range(which, 4):
                if h not in hour_seen:
                    hour_seen.add(h)
                    backtrace(num - 1, hour + int(pow(2, h)), minute, h + 1)
                    hour_seen.remove(h)
            # 枚举 minute 可能，对应 m 是 2 ^ (m - 4)
            for m in range(max(which, 4), 10):  # 注意枚举分该至少从 4 开始
                if m not in minute_seen:
                    minute_seen.add(m)
                    backtrace(num - 1, hour, minute + int(pow(2, m - 4)), m + 1)
                    minute_seen.remove(m)
        backtrace(num, 0, 0, 0)
        return res