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