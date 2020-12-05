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
        if sum(nums) == x:
            return len(nums)
        pre = [0]
        for y in nums:
            pre.append(pre[-1]+y)
        t = sum(nums) - x
        res = float('-inf')
        dic = defaultdict(int)
        for i, y in enumerate(pre):
            if y - t in dic:
                res = max(res, i-dic[y-t])
            if y not in dic:
                dic[y] = i
        return len(nums) - res if res != float('-inf') else -1

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
        print(odd)
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
















































