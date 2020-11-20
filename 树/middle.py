from .base import TreeNode, NTreeNode
from typing import List

class MiddleTopic(TreeNode):
	def listOfDepth(self, tree: TreeNode) -> List[ListNode]:
		'''
		desc: 特定深度节点链表
		给定一棵二叉树，设计一个算法，创建含有某一深度上所有节点的链表
		（比如，若一棵树的深度为 D，则会创建出 D 个链表）。
		返回一个包含所有深度的链表的数组。
		'''
        def dfs(tree):
            if not tree:
                return []
            stack = [tree]
            new_stack = []
            n = len(stack)
            output = []
            tmp = []
            while stack:
                while n>0:
                    node = stack.pop(0)
                    tmp.append(node.val)
                    if node.left:
                        new_stack.append(node.left)
                    if node.right:
                        new_stack.append(node.right)
                    n -= 1
                cur = head = ListNode(None)
                for i in tmp:
                    head.next = ListNode(i)
                    head = head.next
                output.append(cur.next)
                tmp = []
                stack = new_stack
                new_stack = []
                n = len(stack)
            return output
        return dfs(tree)

    def constructMaximumBinaryTree(self, nums: List[int]) -> TreeNode:
    	'''
		desc: 最大二叉树
		给定一个不含重复元素的整数数组。一个以此数组构建的最大二叉树定义如下：
			二叉树的根是数组中的最大元素。
			左子树是通过数组中最大值左边部分构造出的最大二叉树。
			右子树是通过数组中最大值右边部分构造出的最大二叉树。
			通过给定的数组构建最大二叉树，并且输出这个树的根节点。
    	'''
    	def dfs(nums):
            if not nums:
                return None
            sorted_nums = [(nums[i], i) for i in range(len(nums))]
            sorted_nums = sorted(sorted_nums, key=lambda x:x[0], reverse=True)
            merge = TreeNode(sorted_nums[0][0])
            merge.left = dfs(nums[:sorted_nums[0][1]])
            merge.right = dfs(nums[sorted_nums[0][1]+1:])
            return merge
        return dfs(nums)
















