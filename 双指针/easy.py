from base import ListNode

class Solution(ListNode):
    def hasCycle(self, head: ListNode) -> bool:
    	'''
		desc: 环形链表
    	'''
    	fast = head
    	slow = head
    	while head and head.next:
    		fast = fast.next.next
    		slow = slow.next
    		if fast == slow:
    			return True
    	return False

    def strStr(self, haystack: str, needle: str) -> int:
    	'''
		desc: 实现 strStr()
		给定一个 haystack 字符串和一个 needle 字符串，
		在 haystack 字符串中找出 needle 字符串出现的第一个位置 (从0开始)。如果不存在，则返回  -1。
    	'''
    	if len(haystack)<len(needle):
    		return -1
    	if haystack==needle:
    		return 0
    	ans = -1
    	for i in range(len(haystack)-len(needle)+1):
    		if haystack[i:i+len(needle)] == needle:
    			return i
    	return ans

    def reverseVowels(self, s: str) -> str:
        '''
        desc: 反转字符串中的元音字母
        '''
        array = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']
        sList = list(s)
        left, right = 0, len(s)-1
        while left < right:
            if sList[left] in array and sList[right] in array:
                sList[left], sList[right] = sList[right], sList[left]
                left += 1
                right -= 1
            if sList[right] not in array:
                right -= 1
            if sList[left] not in array:
                left += 1
        return ''.join(sList)

    def sortedSquares(self, A: List[int]) -> List[int]:
        '''
        desc: 有序数组的平方 
        给定一个按非递减顺序排序的整数数组 A，
        返回每个数字的平方组成的新数组，要求也按非递减顺序排序。
        [-4,-1,0,3,10]
        [0,1,9,16,100]
        '''
        n = len(A)
        ans = [0] * n
        i, j, pos = 0, n - 1, n - 1
        while i <= j:
            if A[i] * A[i] > A[j] * A[j]:
                ans[pos] = A[i] * A[i]
                i += 1
            else:
                ans[pos] = A[j] * A[j]
                j -= 1
            pos -= 1
        return ans

    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        desc: 合并两个有序数组
        给你两个有序整数数组 nums1 和 nums2，
        请你将 nums2 合并到 nums1 中，使 nums1 成为一个有序数组
        """
        p1 = m - 1
        p2 = n - 1
        p = m + n - 1
        while p1 >= 0 and p2 >= 0:
            if nums1[p1] < nums2[p2]:
                nums1[p] = nums2[p2]
                p2 -= 1
            else:
                nums1[p] =  nums1[p1]
                p1 -= 1
            p -= 1
        nums1[:p2 + 1] = nums2[:p2 + 1]




































