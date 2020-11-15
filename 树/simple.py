from .base import TreeNode, NTreeNode
from typing import List
'''
desc: 树相关简单题
'''
class SimpleTopic(TreeNode):
	"""docstring for SimpleTopic"""
	def preorder(self, root: TreeNode) -> List[int]:
		# 前序二叉树
		if root is None:
			return []
		val = root.value
		left = self.preorder(root.left)
		right = self.preorder(root.right)
		return [val] + left + right

	def midorder(self, root: TreeNode) -> List[int]:
		# 中序二叉树
		if root is None:
			return []
		val = root.value
		left = self.preorder(root.left)
		right = self.preorder(root.right)
		return left + [val] + right

	def endorder(self, root: TreeNode) -> List[int]:
		# 后序二叉树
		if root is None:
			return []
		val = root.value
		left = self.preorder(root.left)
		right = self.preorder(root.right)
		return left + right + [val]

	def n_preorder(self, root: NTreeNode) -> List[int]:
		# N叉树 的 前序遍历
		if root is None:
			return []
		stack = [root]
		output = []
		while stack:
			node = stack.pop()
			output.append(node.value)
			stack.extend(node.children[::-1])
		return output
		
	def isBalanced(self, root: TreeNode) -> bool:
		# 平衡二叉树判定
		def helper(root):
			if root is None:
				return 0
			left = helper(root.left)
			right = helper(root.right)
			if left==-1 or right==-1 or abs(left-right)>1:
				return -1
			else:
				return max(left, right) + 1
		return helper(root)>=0

	def levelOrder(self, root: TreeNode) -> List[List[int]]:
		# 按层输出二叉树，从上到下，从左到右
		if root is None:
			return []
		stack = [root]
		n = len(stack)
		output = []
		tmp = []
		new_stack = []
		while stack:
			while n>0:
				tmp_node = stack.pop(0)
				tmp.append(tmp_node.val)
				if tmp_node.left:
					new_stack.append(tmp_node.left)
				if tmp_node.right:
					new_stack.append(tmp_node.right)
				n -= 1
			stack = new_stack
			n = len(stack)
			output.append(tmp)
			tmp = []
			new_stack = []
		return output

	def increasingBST(self, root: TreeNode) -> TreeNode:
		'''
		desc: 递增顺序查找树
			  给你一个树，请你 按中序遍历 重新排列树，
			  使树中最左边的结点现在是树的根，
			  并且每个结点没有左子结点，只有一个右子结点。
		'''
		def midorder(root):
			# 中序遍历
			if root is None:
				return []
			val = root.val
			left = midorder(root.left)
			right = midorder(root.right)
			return left + [val] + right
        
        cur = head = TreeNode(None)
        for v in midorder(root):
            cur.right = TreeNode(v)
            cur = cur.right
		return head.right

	def kthLargest(self, root: TreeNode, k: int) -> int:
		'''
		desc:  二叉搜索树的第k大节点
				给定一棵二叉搜索树，请找出其中第k大的节点。
		'''
    	def dfs(root):
            if not root: return
            dfs(root.right)
            if self.k == 0: return
            self.k -= 1
            if self.k == 0: self.res = root.val
            dfs(root.left)
		self.k = k
		dfs(root)
		return self.res

	def hasPathSum(self, root: TreeNode, sum: int) -> bool:
    	'''
		desc: 路径总和
			  给定一个二叉树和一个目标和，
			  判断该树中是否存在根节点到叶子节点的路径，
			  这条路径上所有节点值相加等于目标和。
    	'''
    	def dfs(root, target):
    		if root is None:
    			return []

    		if root.left is None and root.right is None and root.val==target:
    			return [[root.val]]
    		ret = []
    		left = dfs(root.left, target-root.val)
    		right = dfs(root.right, target-root.val)
    		for i in left+right:
    			ret.append([root.val]+i)
    	return dfs(root, sum)

    def sumRootToLeaf(self, root: TreeNode) -> int:
    	'''
		desc: 从根到叶的二进制数之和
			  给出一棵二叉树，其上每个结点的值都是 0 或 1 。
			  每一条从根到叶的路径都代表一个从最高有效位开始的二进制数。
			  例如，如果路径为 0 -> 1 -> 1 -> 0 -> 1，那么它表示二进制数 01101，也就是 13
    	'''
    	def dfs(root):
            if root is None:
                return []
            if root.left is None and root.right is None:
                return [[str(root.val)]]
            ret = []
            left = dfs(root.left)
            right = dfs(root.right)
            for i in left + right:
                ret.append([str(root.val)]+i)
            return ret

        def binary2ten(nums):
            sum = 0
            for item in nums:
                item = ''.join(item)
                sum += int('0b'+item, 2)
            return sum 
        return binary2ten(dfs(root))

    def tree2str(self, root: TreeNode) -> str:
    	'''
		desc: 根据二叉树创建字符串
			  你需要采用前序遍历的方式，将一个二叉树转换成一个由括号和整数组成的字符串。
			  空节点则用一对空括号 "()" 表示。而且你需要省略所有不影响字符串与原始二叉树
			  之间的一对一映射关系的空括号对。
    	'''
    	def dfs(root):
            if root is None:
                return "()"
            if not root.left and not root.right:
                return "("+str(root.val)+")"
            left = dfs(root.left)
            right = dfs(root.right)
            if right=='()':
                right = ''
            return "("+str(root.val)+left+right+")"
        return dfs(t)[1:-1]

    def isSymmetric(self, root: TreeNode) -> bool:
    	'''
		desc: 对称的二叉树
			  请实现一个函数，用来判断一棵二叉树是不是对称的。
			  如果一棵二叉树和它的镜像一样，那么它是对称的。
			  例如，二叉树 [1,2,2,3,4,4,3] 是对称的。
    	'''
    	def recur(L, R):
            if not L and not R: return True
            if not L or not R or L.val != R.val: return False
            return recur(L.left, R.right) and recur(L.right, R.left)

        return recur(root.left, root.right) if root else True

    def mergeTrees(self, t1: TreeNode, t2: TreeNode) -> TreeNode:
    	'''
		desc: 合并二叉树
			  给定两个二叉树，想象当你将它们中的一个覆盖到另一个上时，
			  两个二叉树的一些节点便会重叠。
			  你需要将他们合并为一个新的二叉树。合并的规则是如果两个节点重叠，
			  那么将他们的值相加作为节点合并后的新值，
			  否则不为 NULL 的节点将直接作为新二叉树的节点。
    	'''
    	if not t1:
            return t2
        if not t2:
            return t1
        
        merged = TreeNode(t1.val + t2.val)
        merged.left = self.mergeTrees(t1.left, t2.left)
        merged.right = self.mergeTrees(t1.right, t2.right)
        return merged
    







