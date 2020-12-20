from base import ListNode
import collections

class Solution(ListNode):
	def threeSumClosest(self, nums: List[int], target: int) -> int:
		'''
		desc: 最接近的三数之和
		给定一个包括 n 个整数的数组 nums 和 一个目标值 target。
		找出 nums 中的三个整数，使得它们的和与 target 最接近。
		返回这三个数的和。假定每组输入只存在唯一答案。
		'''
		nums.sort()
        ans = nums[0] + nums[1] + nums[2]
        for i in range(len(nums)):
            left = i+1
            right = len(nums)-1
            while left<right:
                sum = nums[left] + nums[right] + nums[i]
                if abs(target-sum) < abs(target-ans):
                    ans = sum
                if sum > target:
                    right -= 1
                elif sum<target:
                    left += 1
                else:
                    return ans
        return ans

    def partition(self, head: ListNode, x: int) -> ListNode:
    	'''
		desc: 分隔链表
		给定一个链表和一个特定值 x，对链表进行分隔，
		使得所有小于 x 的节点都在大于或等于 x 的节点之前。
		你应当保留两个分区中每个节点的初始相对位置。
    	'''
    	node1 = ListNode(None)
        node2 = ListNode(None)
        tmp1 = node1
        tmp2 = node2
        while head:
            if head.val<x:
                node1.next = ListNode(head.val)
                node1 = node1.next
                head = head.next
            else:
                node2.next = ListNode(head.val)
                node2 = node2.next
                head = head.next
        node1.next = tmp2.next
        return tmp1.next

    def totalFruit(self, tree: List[int]) -> int:
        '''
        desc: 水果成篮
        其实就是求一个数组中允许有两种值出现的最长子序列
        '''
        ans = i = 0
        count = collections.Counter()
        for j, x in enumerate(tree):
            count[x] += 1
            while len(count) >= 3:
                count[tree[i]] -= 1
                if count[tree[i]] == 0:
                    del count[tree[i]]
                i += 1
            ans = max(ans, j - i + 1)
        return ans

    def minOperations(self, nums: List[int], x: int) -> int:
        '''
        desc: 将 x 减到 0 的最小操作数
        给你一个整数数组 nums 和一个整数 x 。
        每一次操作时，你应当移除数组 nums 最左边或最右边的元素，
        然后从 x 中减去该元素的值。请注意，需要 修改 数组以供接下来的操作使用。
        如果可以将 x 恰好 减到 0 ，返回 最小操作数 ；否则，返回 -1 。
        '''
        sum_val = sum(nums)
        length = len(nums)
        if sum_val < x:
            return -1
        if sum_val == x:
            return length
        target = sum_val - x
        left, right = 0, 0
        # 累加和
        cur_val = 0
        res = float('inf')
        while right < length:
            cur_val += nums[right]
            while cur_val >= target and left <= right:
                if cur_val == target:
                    res = min(res, length - right + left - 1)
                cur_val -= nums[left]
                left += 1
            right += 1
        return res if res != float('inf') else -1

    def numberOfSubarrays(self, nums: List[int], k: int) -> int:
        '''
        desc: 统计「优美子数组」
        给你一个整数数组 nums 和一个整数 k。
        如果某个 连续 子数组中恰好有 k 个奇数数字，我们就认为这个子数组是「优美子数组」。
        请返回这个数组中「优美子数组」的数目。
        '''
        n = len(nums)
        odd = [-1]
        ans = 0
        for i in range(n):
            if nums[i] % 2 == 1:
                odd.append(i)
        odd.append(n)
        for i in range(1, len(odd) - k):
            ans += (odd[i] - odd[i - 1]) * (odd[i + k] - odd[i + k - 1])
        return ans

    def characterReplacement(self, s: str, k: int) -> int:
        '''
        desc: 替换后的最长重复字符
        给你一个仅由大写英文字母组成的字符串，
        你可以将任意位置上的字符替换成另外的字符，
        总共可最多替换 k 次。在执行上述操作后，找到包含重复字母的最长子串的长度。
        '''
        char_counter = defaultdict(int)  
        left = right = 0 
        max_freq = 0  
        while right < len(s): 
            char_counter[s[right]] += 1
            max_freq = max(max_freq, char_counter[s[right]])
            if (right - left + 1) - max_freq > k: 
                char_counter[s[left]] -= 1 
                left += 1
            right += 1
        else:
            return right - left

    def lengthOfLongestSubstring(self, s: str) -> int:
        '''
        desc: 无重复字符的最长子串
        给定一个字符串，请你找出其中不含有重复字符的 最长子串 的长度。
        '''
        occ = set()
        n = len(s)
        rk, ans = -1, 0
        for i in range(n):
            if i != 0:
                occ.remove(s[i - 1])
            while rk + 1 < n and s[rk + 1] not in occ:
                occ.add(s[rk + 1])
                rk += 1
            ans = max(ans, rk - i + 1)
        return ans

    def intervalIntersection(self, A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
        '''
        desc: 区间列表的交集
        给定两个由一些 闭区间 组成的列表，每个区间列表都是成对不相交的，
        并且已经排序。返回这两个区间列表的交集。
        '''
        ans = []
        i = j = 0
        while i < len(A) and j < len(B):
            lo = max(A[i][0], B[j][0])
            hi = min(A[i][1], B[j][1])
            if lo <= hi:
                ans.append([lo, hi])
            if A[i][1] < B[j][1]:
                i += 1
            else:
                j += 1
        return ans

    def rotateRight(self, head: ListNode, k: int) -> ListNode:
        '''
        desc: 旋转链表
        给定一个链表，旋转链表，将链表每个节点向右移动 k 个位置，其中 k 是非负数。
        '''
        if not head:
            return None
        if not head.next:
            return head
        old_tail = head
        n = 1
        while old_tail.next:
            old_tail = old_tail.next
            n += 1
        old_tail.next = head
        new_tail = head
        for i in range(n - k % n - 1):
            new_tail = new_tail.next
        new_head = new_tail.next
        new_tail.next = None
        return new_head

    def checkPalindromeFormation(self, a: str, b: str) -> bool:
        '''
        desc: 分割两个字符串得到回文串
        '''
        n=len(a)
        if a==a[::-1] or b==b[::-1]:return True
        i,j=0,n-1
        while i<=j and (a[i]==b[j] or b[i]==a[j]):
            i+=1
            j-=1
        if i>=j or a[i:j+1]==a[i:j+1][::-1] or b[i:j+1]==b[i:j+1][::-1]:return True
        return False

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        '''
        desc: 三数之和
        给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，
        使得 a + b + c = 0 ？请你找出所有满足条件且不重复的三元组。
        '''
        n = len(nums)
        nums.sort()
        ans = list()
        for first in range(n):
            if first > 0 and nums[first] == nums[first - 1]:
                continue
            third = n - 1
            target = -nums[first]
            for second in range(first + 1, n):
                if second > first + 1 and nums[second] == nums[second - 1]:
                    continue
                while second < third and nums[second] + nums[third] > target:
                    third -= 1
                if second == third:
                    break
                if nums[second] + nums[third] == target:
                    ans.append([nums[first], nums[second], nums[third]])
        return ans

    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        '''
        desc: 四数之和
        '''
        quadruplets = list()
        if not nums or len(nums) < 4:
            return quadruplets
        nums.sort()
        length = len(nums)
        for i in range(length - 3):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            if nums[i] + nums[i + 1] + nums[i + 2] + nums[i + 3] > target:
                break
            if nums[i] + nums[length - 3] + nums[length - 2] + nums[length - 1] < target:
                continue
            for j in range(i + 1, length - 2):
                if j > i + 1 and nums[j] == nums[j - 1]:
                    continue
                if nums[i] + nums[j] + nums[j + 1] + nums[j + 2] > target:
                    break
                if nums[i] + nums[j] + nums[length - 2] + nums[length - 1] < target:
                    continue
                left, right = j + 1, length - 1
                while left < right:
                    total = nums[i] + nums[j] + nums[left] + nums[right]
                    if total == target:
                        quadruplets.append([nums[i], nums[j], nums[left], nums[right]])
                        while left < right and nums[left] == nums[left + 1]:
                            left += 1
                        left += 1
                        while left < right and nums[right] == nums[right - 1]:
                            right -= 1
                        right -= 1
                    elif total < target:
                        left += 1
                    else:
                        right -= 1
        return quadruplets

    def checkInclusion(self, s1: str, s2: str) -> bool:
        '''
        desc: 字符串的排列
        给定两个字符串 s1 和 s2，写一个函数来判断 s2 是否包含 s1 的排列。
        换句话说，第一个字符串的排列之一是第二个字符串的子串。
        '''
        if len(s1) > len(s2):
            return False
        c1 = {k:0 for k in string.ascii_lowercase}
        cur = {k:0 for k in string.ascii_lowercase}
        for a in s1: c1[a] += 1
        for i in range(len(s2)):
            cur[s2[i]] += 1
            if i >= len(s1):
                cur[s2[i - len(s1)]] -= 1
            if c1 == cur: return True
        return False

    def longestOnes(self, A: List[int], K: int) -> int:
        '''
        desc: 最大连续1的个数 III
        给定一个由若干 0 和 1 组成的数组 A，我们最多可以将 K 个值从 0 变成 1 。
        返回仅包含 1 的最长（连续）子数组的长度。
        '''
        left, right = 0, 0
        count = 0
        for right in range(len(A)): 
            if A[right] == 0:       
                count += 1
            if count > K:           
                if A[left] == 0:    
                    count -= 1
                left += 1           
        return right - left + 1  

    def balancedString(self, s: str) -> int:
        '''
        替换子串得到平衡字符串
        有一个只含有 'Q', 'W', 'E', 'R' 四种字符，且长度为 n 的字符串。
        假如在该字符串中，这四个字符都恰好出现 n/4 次，那么它就是一个「平衡字符串」。
        给你一个这样的字符串 s，请通过「替换一个子串」的方式，使原字符串 s 变成一个「平衡字符串」。
        你可以用和「待替换子串」长度相同的 任何 其他字符串来完成替换。
        请返回待替换子串的最小可能长度。如果原字符串自身就是一个平衡字符串，则返回 0。
        '''
        if not s:
            return 0
        # 统计每个字符的个数
        s_c = dict(Counter(s))
        # 统计哪些字符需要替换，以及需要被替换的个数
        s_replaced = dict()
        for ch, _c in s_c.items():
            if _c > len(s) // 4:
                s_replaced[ch] = _c - len(s) // 4
        # 滑窗左边界
        left = 0
        # 滑窗右边界
        right = 0
        # 表示滑窗内包含的需要被替换的字符及其个数
        origin_chs = {ch: 0 for ch in s_replaced.keys()}
        rst = float('inf')
        while right < len(s):
            ch = s[right]
            if ch not in origin_chs:
                # 如果该字符不在需要替换的滑窗内，跳过，右边界前进，扩大窗口
                right += 1
                continue
            # 找到一个需要替换的字符，相应个数+1
            origin_chs[ch] += 1
            if any([origin_chs[_ch] < s_replaced[_ch] for _ch in origin_chs.keys()]):
                # 当需要被替换的字符有任何一个没有达到替换的个数时，右边界前进，扩大窗口
                right += 1
            else:
                # 都达到个数后，开始处理
                # 将前面预先加上的字符去掉
                origin_chs[ch] -= 1
                # 更新最小值
                rst = min(rst, right - left + 1)
                # 如果将要移出窗口的字符在需要替换的字符中，移出后，相应个数减一
                if s[left] in origin_chs:
                    origin_chs[s[left]] -= 1
                # 左边界前进，缩小窗口
                left += 1
        return 0 if rst == float('inf') else rst

    def longestMountain(self, A: List[int]) -> int:
        '''
        desc: 数组中的最长山脉
        '''
        if len(A)<3:
            return 0
        n = len(A)
        left = [0] * n
        for i in range(1, n):
            left[i] = left[i - 1] + 1 if A[i - 1] < A[i] else 0
        right = [0] * n
        for i in range(n - 2, -1, -1):
            right[i] = right[i + 1] + 1 if A[i + 1] < A[i] else 0
        ans = 0
        for i in range(n):
            if left[i] > 0 and right[i] > 0:
                ans = max(ans, left[i] + right[i] + 1)
        return ans

    def minSubArrayLen(self, s: int, nums: List[int]) -> int:
        '''
        desc: 长度最小的子数组
        给定一个含有 n 个正整数的数组和一个正整数 s ，
        找出该数组中满足其和 ≥ s 的长度最小的 连续 子数组，
        并返回其长度。如果不存在符合条件的子数组，返回 0。
        '''
        if not nums:
            return 0
        n = len(nums)
        ans = n + 1
        start, end = 0, 0
        total = 0
        while end < n:
            total += nums[end]
            while total >= s:
                ans = min(ans, end - start + 1)
                total -= nums[start]
                start += 1
            end += 1
        return 0 if ans == n + 1 else ans

    def permutation(self, s: str) -> List[str]:
        '''
        desc: 字符串的排列
        输入一个字符串，打印出该字符串中字符的所有排列。
        你可以以任意顺序返回这个字符串数组，但里面不能有重复元素。
        '''
        self.res = []
        n = len(s)
        def backtrack(s, path):
            if not s:
                self.res.append(path)
            seen = set()
            for i in range(len(s)):
                if s[i] in seen: continue
                seen.add(s[i])
                backtrack(s[:i]+s[i+1:], path + s[i])
        backtrack(s, "")
        return self.res

    def findLongestWord(self, s: str, d: List[str]) -> str:
        '''
        desc: 通过删除字母匹配到字典里最长单词
        给定一个字符串和一个字符串字典，找到字典里面最长的字符串，
        该字符串可以通过删除给定字符串的某些字符来得到。如果答案不止一个，
        返回长度最长且字典顺序最小的字符串。如果答案不存在，则返回空字符串。
        '''
        res = ""
        for str_d in d:
            j = 0
            for i in range(len(s)):
                if j < len(str_d):
                    if s[i] == str_d[j]: j += 1
                    if j == len(str_d):
                        if (len(str_d)>len(res)) or (len(str_d) == len(res) and str_d<res):
                            res = str_d
        return res

    def numRescueBoats(self, people: List[int], limit: int) -> int:
        '''
        desc: 救生艇
        第 i 个人的体重为 people[i]，每艘船可以承载的最大重量为 limit。
        每艘船最多可同时载两人，但条件是这些人的重量之和最多为 limit。
        返回载到每一个人所需的最小船数。(保证每个人都能被船载)。
        '''
        people.sort()
        i, j = 0, len(people) - 1
        ans = 0
        while i <= j:
            ans += 1
            if people[i] + people[j] <= limit:
                i += 1
            j -= 1
        return ans

    def smallestDifference(self, a: List[int], b: List[int]) -> int:
        '''
        desc: 最小差
        给定两个整数数组a和b，计算具有最小差绝对值的一对数值（每个数组中取一个值），并返回该对数值的差
        '''
        if not a or not b:
            return 0
        a.sort()
        b.sort()
        a_index = 0
        b_index = 0
        diff = float('inf')
        while a_index < len(a) and b_index < len(b):
            diff = min(diff, abs(a[a_index] - b[b_index]))
            if a[a_index] - b[b_index] > 0:
                b_index += 1
            else:
                a_index += 1
        return abs(diff)

    def findClosest(self, words: List[str], word1: str, word2: str) -> int:
        '''
        desc: 单词距离
        有个内含单词的超大文本文件，给定任意两个单词，
        找出在这个文件中这两个单词的最短距离(相隔单词数)。
        如果寻找过程在这个文件中会重复多次，而每次寻找的单词不同，你能对此优化吗?
        '''
        i, ans = 0, len(words)
        for j, word in enumerate(words):
            if word == word1 or word == word2:  # 遇到两个词之一
                if word != words[i] and (words[i] == word1 or words[i] == word2):
                    ans = min(ans, j - i) 
                i = j       # 每次都更新 i即可指向word1也可指向word2               
        return ans

    def maximumUniqueSubarray(self, nums: List[int]) -> int:
        '''
        desc: 删除子数组的最大得分
        给你一个正整数数组 nums ，请你从中删除一个含有 若干不同元素 的子数组。删除子数组的 得分 就是子数组各元素之 和 。
        返回 只删除一个 子数组可获得的 最大得分 。
        如果数组 b 是数组 a 的一个连续子序列，即如果它等于 a[l],a[l+1],...,a[r] ，那么它就是 a 的一个子数组。
        '''
        pre_num = [0]
        for num in nums:
            pre_num.append(pre_num[-1] + num)
        start, pos_dict, ans = 0, {}, 0
        for i, n in enumerate(nums):
            if n in pos_dict and pos_dict[n] >= start:
                ans = max(ans, pre_num[i] - pre_num[start])
                start = pos_dict[n] + 1
            pos_dict[n] = i
        return max(ans, pre_num[-1] - pre_num[start])

















